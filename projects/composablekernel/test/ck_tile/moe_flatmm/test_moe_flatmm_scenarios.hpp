// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "ck_tile/core.hpp"

// Shared MoE FlatMM scenario set, exercised against every dtype/MoeKindTag
// instantiation of TestMoeFlatmmBase / TestMXMoeFlatmmBase / TestA16W4MoeFlatmmBase.
//
// Each scenario maps to a specific moe_sorting layout regime so a regression
// can be pinned to "what shape made it fail":
//
//   Typical              - mixed per-tile densities; production-realistic
//                          (`num_tokens=64, topk=2, experts=8`).
//   SingleExpertDense    - one expert, multiple dense tiles, no sentinels in
//                          the body of the routing.
//   AllExpertsPopulated  - high topk so every expert ends up with multiple
//                          slices (`topk=4, experts=8`).
//   OneTokenPerTile      - sparse: each expert gets exactly one real token
//                          and (M_a-1) sentinels per tile. This is the regime
//                          the original hand-rolled fixture inadvertently
//                          tested; kept here as a control.
//   FullTileNoSentinels  - exactly M_a tokens routed to one expert via a
//                          forced topk-id of all zero; produces a single
//                          tile with no sentinels.
//   EmptyExperts         - more experts than active routings so most experts
//                          carry zero tokens (skipped by reference_moe_sorting,
//                          matching production moe_sorting() defaults).
//   LargeK               - shape-coverage knob: long K to flush prefetch and
//                          stress the K-loop unroll.
//
// Test files instantiate these via `MOE_FLATMM_DECLARE_SCENARIOS(SuiteName)`,
// which expands to one `TYPED_TEST(SuiteName, <scenario>)` per scenario.

namespace test_moe_flatmm {

template <typename Fixture>
inline void run_typical(Fixture& f)
{
    f.run_test(/*num_tokens=*/64, /*topk=*/2, /*experts=*/8, /*N=*/512, /*K=*/256);
}

template <typename Fixture>
inline void run_single_expert_dense(Fixture& f)
{
    f.run_test(/*num_tokens=*/64, /*topk=*/1, /*experts=*/1, /*N=*/512, /*K=*/256);
}

template <typename Fixture>
inline void run_all_experts_populated(Fixture& f)
{
    f.run_test(/*num_tokens=*/64, /*topk=*/4, /*experts=*/8, /*N=*/512, /*K=*/256);
}

template <typename Fixture>
inline void run_one_token_per_tile(Fixture& f)
{
    f.run_test(/*num_tokens=*/8, /*topk=*/1, /*experts=*/8, /*N=*/512, /*K=*/256);
}

template <typename Fixture>
inline void run_full_tile_no_sentinels(Fixture& f)
{
    constexpr int M_a = Fixture::FlatmmConfig::M_Tile;
    std::vector<ck_tile::index_t> forced(static_cast<std::size_t>(M_a), 0);
    f.run_test(
        /*num_tokens=*/M_a, /*topk=*/1, /*experts=*/1, /*N=*/512, /*K=*/256, std::move(forced));
}

template <typename Fixture>
inline void run_empty_experts(Fixture& f)
{
    f.run_test(/*num_tokens=*/4, /*topk=*/1, /*experts=*/16, /*N=*/512, /*K=*/256);
}

template <typename Fixture>
inline void run_large_k(Fixture& f)
{
    f.run_test(/*num_tokens=*/64, /*topk=*/2, /*experts=*/4, /*N=*/512, /*K=*/2048);
}

} // namespace test_moe_flatmm

// One TYPED_TEST per scenario. Suite-name placement keeps the gtest filter
// (`--gtest_filter=*Typical*` etc.) intuitive across dtype suites.
#define MOE_FLATMM_DECLARE_SCENARIOS(SuiteName)                                                   \
    TYPED_TEST(SuiteName, Typical) { test_moe_flatmm::run_typical(*this); }                       \
    TYPED_TEST(SuiteName, SingleExpertDense) { test_moe_flatmm::run_single_expert_dense(*this); } \
    TYPED_TEST(SuiteName, AllExpertsPopulated)                                                    \
    {                                                                                             \
        test_moe_flatmm::run_all_experts_populated(*this);                                        \
    }                                                                                             \
    TYPED_TEST(SuiteName, OneTokenPerTile) { test_moe_flatmm::run_one_token_per_tile(*this); }    \
    TYPED_TEST(SuiteName, FullTileNoSentinels)                                                    \
    {                                                                                             \
        test_moe_flatmm::run_full_tile_no_sentinels(*this);                                       \
    }                                                                                             \
    TYPED_TEST(SuiteName, EmptyExperts) { test_moe_flatmm::run_empty_experts(*this); }            \
    TYPED_TEST(SuiteName, LargeK) { test_moe_flatmm::run_large_k(*this); }
