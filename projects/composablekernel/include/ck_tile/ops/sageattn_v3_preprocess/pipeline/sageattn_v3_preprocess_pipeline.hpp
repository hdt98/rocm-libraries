// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_mxfp4_pack.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_problem.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_policy.hpp"

namespace ck_tile {

template <typename Problem_,
          typename Policy_ = SA3QKPrepPolicy<Problem_>>
struct SA3QKPrepPipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;
    using InputT  = typename Problem::InputT;

    static constexpr index_t kRows             = Problem::kRows;
    static constexpr index_t kCols             = Problem::kCols;
    static constexpr index_t kScaleGranularity = Problem::kScaleGranularity;
    static constexpr index_t kBlockSize        = Problem::kBlockSize;
    static constexpr index_t kGroups           = Problem::kGroups;
    static constexpr index_t kG                = Problem::kG;
    static constexpr index_t kLoadVec          = Problem::kLoadVec;
    static constexpr index_t kVecPerG          = Problem::kVecPerG;
    static constexpr index_t kThreadsPerCol    = Problem::kThreadsPerCol;

    static constexpr index_t kLdsPad         = Policy::kLdsPad;
    static constexpr index_t kLdsRowStride   = Policy::kLdsRowStride;
    static constexpr index_t kQTileBytes     = Policy::kQTileBytes;
    static constexpr index_t kSmemMeanOffset = Policy::kSmemMeanOffset;
    static constexpr index_t kSmemMeanBytes  = Policy::kSmemMeanBytes;

    // K hat shuffle parameters for FMHA async K load (warp-interleaved LDS layout).
    // pk_fp4_t: KVector=16, kK0 depends on MFMA shape.
    static constexpr index_t kFwdK0         = (kCols <= 128) ? 64 : 128;
    static constexpr index_t kFwdKVector    = 16;
    static constexpr index_t kFwdLanesPerK  = kFwdK0 / kFwdKVector;
    static constexpr index_t kFwdLaneGroups = 64 / kFwdLanesPerK;
    static constexpr index_t kFwdNumWarps   = 4;
    static constexpr index_t kShuffleBlock  = kFwdLaneGroups * kFwdNumWarps;

    CK_TILE_DEVICE static constexpr index_t shuffle_row(index_t row)
    {
        index_t blk = row / kShuffleBlock;
        index_t loc = row % kShuffleBlock;
        return blk * kShuffleBlock + (loc % kFwdLaneGroups) * kFwdNumWarps +
               loc / kFwdLaneGroups;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Policy::GetSmemSize(); }
    CK_TILE_HOST_DEVICE static constexpr index_t GetKSmemSize() { return Policy::GetKSmemSize(); }

    CK_TILE_DEVICE void RunLoadQTile(const InputT* __restrict__ q_ptr, void* smem) const
    {
        InputT* smem_q = reinterpret_cast<InputT*>(smem);

        constexpr auto dstr      = Policy::MakeQKTileDstr();
        const auto q_global_view = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(static_cast<index_t>(kCols), number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto q_global_win = make_tile_window(
            q_global_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto q_tile = load_tile(q_global_win);

        auto smem_q_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_q,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(number<kLdsRowStride>{}, number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto smem_q_win = make_tile_window(
            smem_q_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        store_tile(smem_q_win, q_tile);
    }

    CK_TILE_DEVICE void RunQMean(void* smem,
                                 InputT* __restrict__ q_mean_ptr,
                                 index_t n_rows_valid) const
    {
        const InputT* smem_q = reinterpret_cast<const InputT*>(smem);
        float* smem_mean =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + kSmemMeanOffset);

        constexpr index_t kWarps_      = kBlockSize / 64;
        constexpr index_t kColsPerWarp = kCols / kWarps_;
        const index_t tid              = get_thread_id();
        const index_t warp_id          = tid / 64;
        const index_t lane_id          = tid % 64;
        const index_t col_idx          = warp_id * kColsPerWarp + lane_id / kThreadsPerCol;
        const index_t r_id             = lane_id % kThreadsPerCol;
        float acc                      = 0.0f;

        for(index_t r = r_id; r < n_rows_valid; r += kThreadsPerCol)
            acc += static_cast<float>(smem_q[r * kLdsRowStride + col_idx]);

        auto acc_tile = make_static_distributed_tensor<float>(Policy::MakeMeanReduceTileDstr());
        acc_tile.get_thread_buffer()(number<0>{}) = acc;
        block_tile_reduce_xor_sync(acc_tile, [](float a, float b) { return a + b; });

        if(r_id == 0)
        {
            const float mean    = acc_tile.get_thread_buffer()[number<0>{}] /
                                  static_cast<float>(n_rows_valid);
            smem_mean[col_idx]  = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
    }

    CK_TILE_DEVICE void RunQQuantize(const void* smem,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr,
                                     index_t n_rows_valid) const
    {
        const InputT* smem_q = reinterpret_cast<const InputT*>(smem);
        const float* smem_mean =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(smem) +
                                           kSmemMeanOffset);
        const index_t tid            = get_thread_id();
        constexpr index_t kNumGroups = kGroups;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        constexpr auto dstr   = Policy::MakeQKTileDstr();
        const auto q_lds_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_q,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(number<kLdsRowStride>{}, number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto q_lds_win = make_tile_window(
            q_lds_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto q_tile = load_tile(q_lds_win);

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        static_for<0, kG, 1>{}([&](auto j) {
            const float val = static_cast<float>(q_tile.get_thread_buffer()[j]) -
                              smem_mean[d_start + j];
            group_data[j]   = val;
            max_abs         = max(max_abs, abs(val));
        });

        const float scale = bit_cast<float>(
            (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
            numeric_traits<float>::head_mask);

        dst_scale_ptr[row_idx * kNumGroups + grp_idx] =
            static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

        PackFP4Group<kScaleGranularity>(group_data,
                                        dst_hat_ptr + row_idx * (kCols / 2) +
                                            grp_idx * (kScaleGranularity / 2),
                                        scale);
    }

    CK_TILE_DEVICE void RunKSmoothAndQuantize(const InputT* __restrict__ src_ptr,
                                              const float* __restrict__ k_mean_float,
                                              float seqlen_k_inv,
                                              InputT* __restrict__ k_prime_ptr,
                                              index_t k_prime_stride,
                                              uint8_t* __restrict__ dst_hat_ptr,
                                              uint8_t* __restrict__ dst_scale_ptr,
                                              void* smem,
                                              index_t n_rows_valid) const
    {
        const index_t tid = get_thread_id();

        float* smem_f = reinterpret_cast<float*>(smem);
        for(index_t d = tid; d < kCols; d += kBlockSize)
            smem_f[d] = k_mean_float[d] * seqlen_k_inv;
        block_sync_lds();

        constexpr index_t kNumGroups = kGroups;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        const index_t srow = shuffle_row(row_idx);

        if(row_idx >= n_rows_valid)
        {
            InputT* dst_row = k_prime_ptr + row_idx * k_prime_stride + d_start;
            for(index_t j = 0; j < kScaleGranularity; j++)
                dst_row[j] = InputT{0};
            dst_scale_ptr[srow * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + srow * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        constexpr auto dstr   = Policy::MakeQKTileDstr();
        const auto k_src_view = make_naive_tensor_view<address_space_enum::global>(
            src_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(static_cast<index_t>(kCols), number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto k_src_win = make_tile_window(
            k_src_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto k_tile = load_tile(k_src_win);

        float group_data[kScaleGranularity];
        float max_abs    = 0.0f;
        auto kprime_tile = make_static_distributed_tensor<InputT>(dstr);

        static_for<0, kG, 1>{}([&](auto j) {
            const float k_val                  = static_cast<float>(k_tile.get_thread_buffer()[j]);
            const float centered               = k_val - smem_f[d_start + j];
            group_data[j]                      = centered;
            max_abs                            = max(max_abs, abs(centered));
            kprime_tile.get_thread_buffer()(j) = static_cast<InputT>(centered);
        });

        const auto k_dst_view = make_naive_tensor_view<address_space_enum::global>(
            k_prime_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(k_prime_stride, number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto k_dst_win = make_tile_window(
            k_dst_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        store_tile(k_dst_win, kprime_tile);

        const float scale = bit_cast<float>(
            (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
            numeric_traits<float>::head_mask);

        dst_scale_ptr[srow * kNumGroups + grp_idx] =
            static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

        PackFP4Group<kScaleGranularity>(group_data,
                                        dst_hat_ptr + srow * (kCols / 2) +
                                            grp_idx * (kScaleGranularity / 2),
                                        scale);
    }
};

template <typename Problem_,
          typename Policy_ = SA3KMeanPolicy<Problem_>>
struct SA3KMeanPipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;
    using InputT  = typename Problem::InputT;

    static constexpr index_t kCols         = Problem::kCols;
    static constexpr index_t kBlockSize    = Problem::kBlockSize;
    static constexpr index_t kVec          = Problem::kVec;
    static constexpr index_t kColsPerWarp  = Problem::kColsPerWarp;
    static constexpr index_t kRowsPerGroup = Problem::kRowsPerGroup;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Policy::GetSmemSize(); }

    CK_TILE_DEVICE void RunKMean(const InputT* __restrict__ k_chunk,
                                 index_t stride_k,
                                 index_t num_rows,
                                 float* __restrict__ k_mean_out) const
    {
        constexpr auto acc_dstr = Policy::MakeAccDstr();
        constexpr auto k_dstr   = Policy::MakeKLoadDstr();

        auto acc = make_static_distributed_tensor<float>(acc_dstr);
        clear_tile(acc);

        const auto k_view = make_naive_tensor_view<address_space_enum::global>(
            k_chunk,
            make_tuple(num_rows, number<kCols>{}),
            make_tuple(stride_k, number<1>{}),
            number<kVec>{},
            number<1>{});
        auto k_window = make_tile_window(
            k_view,
            make_tuple(number<kRowsPerGroup>{}, number<kCols>{}),
            {0, 0},
            k_dstr);

        const index_t num_full = num_rows / kRowsPerGroup;
        const index_t tail     = num_rows % kRowsPerGroup;

        for(index_t tile = 0; tile < num_full; tile++)
        {
            const auto k_tile = load_tile(k_window);
            tile_elementwise_inout(
                [](float& a, const InputT& b) { a += type_convert<float>(b); }, acc, k_tile);
            move_tile_window(k_window, {kRowsPerGroup, 0});
        }

        if(tail > 0)
        {
            const index_t lane_id   = get_lane_id();
            const index_t row_grp   = lane_id % kRowsPerGroup;
            const index_t warp_id   = get_warp_id();
            const index_t col_grp_l = lane_id / kRowsPerGroup;
            const index_t col_base  = (warp_id * kColsPerWarp + col_grp_l) * kVec;

            for(index_t r = 0; r < tail; r++)
            {
                if(row_grp == r)
                {
                    const index_t abs_row = num_full * kRowsPerGroup + r;
                    const InputT* row_ptr = k_chunk + abs_row * stride_k + col_base;
                    static_for<0, kVec, 1>{}([&](auto j) {
                        acc.get_thread_buffer()(j) += type_convert<float>(row_ptr[j.value]);
                    });
                }
            }
        }

        block_tile_reduce_xor_sync(acc, [](float a, float b) { return a + b; });

        const index_t lane_id_f  = get_lane_id();
        const index_t row_grp_f  = lane_id_f % kRowsPerGroup;
        const index_t warp_id_f  = get_warp_id();
        const index_t col_grp_lf = lane_id_f / kRowsPerGroup;
        const index_t col_base_f = (warp_id_f * kColsPerWarp + col_grp_lf) * kVec;

        if(row_grp_f == 0)
        {
            static_for<0, kVec, 1>{}([&](auto j) {
                __hip_atomic_fetch_add(&k_mean_out[col_base_f + j.value],
                                       acc.get_thread_buffer()[j],
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
            });
        }
    }
};

template <typename Problem_,
          typename Policy_ = SA3VPrepPolicy<Problem_>>
struct SA3VPrepPipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;
    using InputT  = typename Problem::InputT;

    static constexpr index_t kVHdimTile       = Problem::kVHdimTile;
    static constexpr index_t kVGroup          = Problem::kVGroup;
    static constexpr index_t kBlockSize       = Problem::kBlockSize;
    static constexpr index_t kLoadVec         = Problem::kLoadVec;
    static constexpr index_t kVGroupsPerBlock = Problem::kVGroupsPerBlock;

    // V hat shuffle: same warp-interleaved logic as K, applied to hdim_v (N) dimension.
    static constexpr index_t kFwdK1         = (kVHdimTile <= 128) ? 64 : 128;
    static constexpr index_t kFwdKVector    = 16;
    static constexpr index_t kFwdLanesPerK  = kFwdK1 / kFwdKVector;
    static constexpr index_t kFwdLaneGroups = 64 / kFwdLanesPerK;
    static constexpr index_t kFwdNumWarps   = 4;
    static constexpr index_t kShuffleBlock  = kFwdLaneGroups * kFwdNumWarps;

    CK_TILE_DEVICE static constexpr index_t shuffle_col(index_t col)
    {
        index_t blk = col / kShuffleBlock;
        index_t loc = col % kShuffleBlock;
        return blk * kShuffleBlock + (loc % kFwdLaneGroups) * kFwdNumWarps +
               loc / kFwdLaneGroups;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Policy::GetSmemSize(); }

    CK_TILE_DEVICE void RunLoadToSmem(const InputT* __restrict__ v_base,
                                      index_t hdim,
                                      index_t seqlen_k,
                                      index_t seqlen_k_real,
                                      index_t g_abs_base,
                                      InputT* smem_v) const
    {
        constexpr auto sg_load_dstr = Policy::MakeSingleGroupLoadDstr();

        const auto v_global_view = make_naive_tensor_view<address_space_enum::global>(
            v_base,
            make_tuple(seqlen_k, number<kVHdimTile>{}),
            make_tuple(hdim, number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto v_global_win = make_tile_window(
            v_global_view,
            make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
            {g_abs_base * kVGroup, 0},
            sg_load_dstr);

        auto smem_v_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_v,
            make_tuple(number<kVGroupsPerBlock * kVGroup>{}, number<kVHdimTile>{}),
            make_tuple(number<kVHdimTile>{}, number<1>{}),
            number<kLoadVec>{},
            number<1>{});
        auto smem_v_win = make_tile_window(
            smem_v_view,
            make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
            {0, 0},
            sg_load_dstr);

        static_for<0, kVGroupsPerBlock, 1>{}([&](auto i_c) {
            auto v_tile = load_tile(v_global_win);

            const index_t sk_group_start = (g_abs_base + i_c.value) * kVGroup;
            if(sk_group_start >= seqlen_k_real)
            {
                tile_elementwise_inout([](InputT& v) { v = InputT{0}; }, v_tile);
            }

            store_tile(smem_v_win, v_tile);
            move_tile_window(v_global_win, {kVGroup, 0});
            move_tile_window(smem_v_win, {kVGroup, 0});
        });
    }

    CK_TILE_DEVICE void RunQuantize(const InputT* smem_v,
                                    index_t g_abs_base,
                                    uint8_t* __restrict__ v_hat_base,
                                    uint8_t* __restrict__ v_scale_base,
                                    index_t stride_v_hat,
                                    index_t stride_v_scale) const
    {
        const index_t tid       = get_thread_id();
        const index_t group_id  = tid / kVHdimTile;
        const index_t col       = tid % kVHdimTile;
        const index_t scol      = shuffle_col(col);
        const index_t row_start = group_id * kVGroup;

        constexpr float rcp_dst_max = 1.0f / 6.0f;
        float group_data[kVGroup];
        float max_abs = 0.0f;

        for(index_t j = 0; j < kVGroup; j++)
        {
            const float val = static_cast<float>(smem_v[(row_start + j) * kVHdimTile + col]);
            group_data[j]   = val;
            max_abs         = max(max_abs, abs(val));
        }

        const float scale = bit_cast<float>(
            (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
            numeric_traits<float>::head_mask);

        const index_t g_abs = g_abs_base + group_id;

        v_scale_base[scol * stride_v_scale + g_abs] =
            static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

        PackFP4Group<kVGroup>(
            group_data,
            v_hat_base + scol * stride_v_hat + g_abs * (kVGroup / 2),
            scale);
    }
};

} // namespace ck_tile
