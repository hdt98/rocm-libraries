// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_moe_sorting.hpp"

namespace test_moe_flatmm {

// Result of `make_moe_sorted_inputs`. All host tensors carry the layout that
// the production `moe_sorting()` kernel emits (and that
// `ck_tile::reference_moe_sorting` reproduces): `sorted_token_ids` packs
// `(token_id, topk_id)` via MOE_SORTING_MOCK_ID, sentinel slots set
// `sorted_expert_weight` to 0, and `sorted_expert_ids` carries the global
// expert id per tile (skipping experts with no assigned tokens).
//
// `M == num_tokens_post_padded == sorted_tile_num * unit_size` is what the
// MoE FlatMM kernel and `reference_moe_gemm_gpu` consume as the M dimension.
template <typename WeightType, typename IndexType>
struct MoeSortedInputs
{
    ck_tile::HostTensor<IndexType> sorted_token_ids;
    ck_tile::HostTensor<WeightType> sorted_expert_weight;
    ck_tile::HostTensor<IndexType> sorted_expert_ids;
    // Only `max_token_id[0] == num_tokens_post_padded` is read by the kernel /
    // reference, but the original test fixtures kept a `[1+i] = i` decoration
    // so we mirror that shape for drop-in compatibility.
    ck_tile::HostTensor<IndexType> max_token_id;
    ck_tile::HostTensor<IndexType> topk_ids;
    ck_tile::HostTensor<WeightType> topk_weights;
    ck_tile::index_t num_tokens_post_padded;
    ck_tile::index_t M;
    ck_tile::index_t sorted_tile_num;
};

// Same routing-id generator the moe_sorting tests use: per-token unique experts
// across the topk dimension. Kept inline so this header is self-contained.
template <typename IndexType>
inline void
moe_topid_unique_gen(std::vector<IndexType>& dst, int tokens, int topk, int num_experts, int seed)
{
    const std::size_t total = static_cast<std::size_t>(topk) * tokens;
    dst.resize(total);
    std::mt19937 rng{static_cast<unsigned int>(seed)};
    std::set<IndexType> unique_set;
    for(std::size_t i = 0; i < total; ++i)
    {
        if(i % topk == 0)
            unique_set.clear();
        IndexType v = static_cast<IndexType>(rng() % num_experts);
        while(unique_set.find(v) != unique_set.end())
            v = static_cast<IndexType>(rng() % num_experts);
        unique_set.insert(v);
        dst[i] = v;
    }
}

// Build production-shaped MoE FlatMM inputs by running the host reference of
// `moe_sorting` over a generated (or forced) routing.
//
//   num_tokens / topk / num_experts / unit_size: standard moe_sorting params
//                                                (unit_size == kernel MPerBlock)
//   seed:           RNG seed for routing + weights
//   forced_topk_ids: optional (num_tokens * topk) routing override; lets edge
//                    cases (single-expert dense, full-tile no-sentinel, ...)
//                    be expressed deterministically.
//   skip_experts_with_zero_token: forwarded to reference_moe_sorting; default
//                                 matches `moe_sorting()`.
template <typename WeightType = float, typename IndexType = ck_tile::index_t>
inline MoeSortedInputs<WeightType, IndexType>
make_moe_sorted_inputs(int num_tokens,
                       int topk,
                       int num_experts,
                       int unit_size,
                       int seed,
                       std::optional<std::vector<IndexType>> forced_topk_ids = std::nullopt,
                       bool skip_experts_with_zero_token                     = true)
{
    if(num_tokens <= 0 || topk <= 0 || num_experts <= 0 || unit_size <= 0)
        throw std::runtime_error("make_moe_sorted_inputs: invalid problem size");
    if(!forced_topk_ids.has_value() && topk > num_experts)
        throw std::runtime_error(
            "make_moe_sorted_inputs: topk must be <= num_experts when forced_topk_ids "
            "is not provided because generated routing requires unique experts per token");

    // Worst-case sorted-output buffer; matches `moe_sorting_api`'s allocation
    // formula: every (token, topk) lands on a different expert, plus per-expert
    // tail padding to a unit_size boundary.
    const int max_output_ids = ck_tile::integer_least_multiple(
        num_tokens * topk + num_experts * unit_size - topk, unit_size);

    // topk_ids: [num_tokens, topk]
    ck_tile::HostTensor<IndexType> topk_ids({num_tokens, topk}, {topk, 1});
    if(forced_topk_ids.has_value())
    {
        if(static_cast<int>(forced_topk_ids->size()) != num_tokens * topk)
            throw std::runtime_error("make_moe_sorted_inputs: forced_topk_ids size mismatch");
        std::copy(forced_topk_ids->begin(), forced_topk_ids->end(), topk_ids.begin());
    }
    else
    {
        moe_topid_unique_gen<IndexType>(topk_ids.mData, num_tokens, topk, num_experts, seed);
    }

    // topk_weights: [num_tokens, topk] in [0, 1/topk) so gemm2's MulRoutedWeight
    // accumulator stays bounded (sum over topk <= 1, mimicking softmax-normalized
    // routing weights in production). Wider ranges cause bf16 to drift outside
    // the 1e-2 tolerance on Gemm2/AllExpertsPopulated. Use a separate RNG stream
    // from routing so weights don't correlate with expert ids.
    ck_tile::HostTensor<WeightType> topk_weights({num_tokens, topk}, {topk, 1});
    {
        std::mt19937 rng{static_cast<unsigned int>(seed) ^ 0x9e3779b9u};
        const float weight_max = 1.f / static_cast<float>(topk);
        std::uniform_real_distribution<float> dist{0.f, weight_max};
        for(auto& v : topk_weights.mData)
            v = static_cast<WeightType>(dist(rng));
    }

    // local_expert_mask is unused (we never enable masking in MoE FlatMM tests).
    ck_tile::HostTensor<IndexType> local_expert_mask({1});

    ck_tile::HostTensor<IndexType> sorted_ids_full({static_cast<std::size_t>(max_output_ids)}, {1});
    ck_tile::HostTensor<WeightType> sorted_weights_full({static_cast<std::size_t>(max_output_ids)},
                                                        {1});
    ck_tile::HostTensor<IndexType> sorted_expert_ids_full(
        {static_cast<std::size_t>(max_output_ids / unit_size)}, {1});

    ck_tile::index_t num_tokens_post_padded = 0;
    ck_tile::reference_moe_sorting<WeightType, IndexType>(topk_ids,
                                                          topk_weights,
                                                          local_expert_mask,
                                                          sorted_ids_full,
                                                          sorted_weights_full,
                                                          sorted_expert_ids_full,
                                                          num_tokens_post_padded,
                                                          num_experts,
                                                          unit_size,
                                                          num_tokens,
                                                          /*local_expert_masking=*/false,
                                                          skip_experts_with_zero_token);

    if(num_tokens_post_padded <= 0 || num_tokens_post_padded > max_output_ids ||
       (num_tokens_post_padded % unit_size) != 0)
    {
        throw std::runtime_error("make_moe_sorted_inputs: reference_moe_sorting returned an "
                                 "invalid num_tokens_post_padded");
    }

    const ck_tile::index_t sorted_tile_num = num_tokens_post_padded / unit_size;

    ck_tile::HostTensor<IndexType> sorted_token_ids(
        {static_cast<std::size_t>(num_tokens_post_padded)}, {1});
    std::copy(sorted_ids_full.begin(),
              sorted_ids_full.begin() + num_tokens_post_padded,
              sorted_token_ids.begin());

    ck_tile::HostTensor<WeightType> sorted_expert_weight(
        {static_cast<std::size_t>(num_tokens_post_padded)}, {1});
    std::copy(sorted_weights_full.begin(),
              sorted_weights_full.begin() + num_tokens_post_padded,
              sorted_expert_weight.begin());

    ck_tile::HostTensor<IndexType> sorted_expert_ids({static_cast<std::size_t>(sorted_tile_num)},
                                                     {1});
    std::copy(sorted_expert_ids_full.begin(),
              sorted_expert_ids_full.begin() + sorted_tile_num,
              sorted_expert_ids.begin());

    // Mirror the original test fixture's `[0]=N, [1+i]=i` decoration for
    // drop-in compatibility, even though the kernel only reads `[0]`.
    ck_tile::HostTensor<IndexType> max_token_id({static_cast<std::size_t>(1 + sorted_tile_num)},
                                                {1});
    max_token_id.mData.assign(static_cast<std::size_t>(1 + sorted_tile_num), 0);
    max_token_id.mData[0] = num_tokens_post_padded;
    for(ck_tile::index_t i = 0; i < sorted_tile_num; ++i)
        max_token_id.mData[1 + i] = i;

    return MoeSortedInputs<WeightType, IndexType>{std::move(sorted_token_ids),
                                                  std::move(sorted_expert_weight),
                                                  std::move(sorted_expert_ids),
                                                  std::move(max_token_id),
                                                  std::move(topk_ids),
                                                  std::move(topk_weights),
                                                  num_tokens_post_padded,
                                                  num_tokens_post_padded,
                                                  sorted_tile_num};
}

} // namespace test_moe_flatmm
