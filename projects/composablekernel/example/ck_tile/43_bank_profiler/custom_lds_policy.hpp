// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Custom LDS policies for bank conflict experimentation.
//
// Each policy inherits from UniversalGemmBasePolicy and overrides only
// the LDS descriptor methods, keeping everything else (block gemm,
// tile distributions, etc.) identical to the default.
//
// Usage in a BankProfileConfig:
//   struct MyConfig : BankProfileConfigBase {
//       ...
//       using LdsPolicy = PaddedLdsPolicy;  // or XorLdsPolicy (default)
//   };

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

// ---------------------------------------------------------------------------
// Tag type: "use whatever the pipeline's default policy is"
// ---------------------------------------------------------------------------
struct DefaultLdsPolicyTag
{
};

// ---------------------------------------------------------------------------
// XorLdsPolicy — the standard XOR swizzle policy (same as production).
// Provided here so configs can reference it explicitly.
// ---------------------------------------------------------------------------
struct XorLdsPolicy : public UniversalGemmBasePolicy<XorLdsPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return UniversalGemmPipelineAgBgCrPolicy{}.template GetBlockGemm<Problem>();
    }
};

// ---------------------------------------------------------------------------
// PaddedLdsPolicy — classic (M+1)*8 / (N+1)*8 padding, NO XOR swizzle.
//
// This is the original GemmPipelineAGmemBGmemCRegV1DefaultPolicy layout:
//   A: shape [K/8, M, 8], stride [(M+1)*8, 8, 1]
//   B: shape [K/8, N, 8], stride [(N+1)*8, 8, 1]
//
// Useful as a baseline to compare XOR vs padding conflict patterns.
// ---------------------------------------------------------------------------
struct PaddedLdsPolicy : public UniversalGemmBasePolicy<PaddedLdsPolicy>
{
    template <typename Problem, typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kMPerBlock>{}, number<8>{}),
            make_tuple(number<(kMPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem, typename OverrideBDataType = remove_cvref_t<typename Problem::BDataType>>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kNPerBlock>{}, number<8>{}),
            make_tuple(number<(kNPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return UniversalGemmPipelineAgBgCrPolicy{}.template GetBlockGemm<Problem>();
    }
};

} // namespace ck_tile
