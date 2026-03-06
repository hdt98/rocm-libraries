// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockAttentionQuantScaleEnum
{
    NO_SCALE      = 0,
    PERTENSOR     = 1,
    BLOCKSCALE    = 2,
    KV_BLOCKSCALE = 3, // Q per-tensor, K/V per-page block scale
    MX            = 4, // Microscaling
    SAGEATTN_V3   = 5, // SageAttention V3: MXFP4 with delta_s smoothing
};

template <BlockAttentionQuantScaleEnum>
struct BlockAttentionQuantScaleEnumToStr;

template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::NO_SCALE>
{
    static constexpr const char* name = "";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::PERTENSOR>
{
    static constexpr const char* name = "pertensor";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::BLOCKSCALE>
{
    static constexpr const char* name = "blockscale";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::KV_BLOCKSCALE>
{
    static constexpr const char* name = "kv_blockscale";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::MX>
{
    static constexpr const char* name = "mx";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::SAGEATTN_V3>
{
    static constexpr const char* name = "sageattnv3";
};

} // namespace ck_tile
