// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

// Intentional duplication of the MoE FlatMM host-side config/type surface from
// `example/ck_tile/18_flatmm/moe_flatmm.hpp`.
//
// The tests should not depend on example-private include paths, so the minimal
// config/types they exercise live locally under `test/ck_tile/moe_flatmm/`.

template <typename DataType>
struct FlatmmConfig32
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 32;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool TiledMMAPermuteN = false; // disable PermuteN when NWarpTile != 16
};

template <typename DataType>
struct FlatmmConfig32_950 : public FlatmmConfig32<DataType>
{
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 64;
};

template <typename DataType>
struct FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 64;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

template <typename DataType>
struct FlatmmConfig16_950 : public FlatmmConfig16<DataType>
{
    static constexpr ck_tile::index_t N_Tile      = 256;
    static constexpr ck_tile::index_t K_Tile      = 256 / sizeof(DataType);
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 128;
    static constexpr int kBlockPerCu              = 1;

    static constexpr int N_Repeat =
        N_Tile / FlatmmConfig16<DataType>::N_Warp_Tile / FlatmmConfig16<DataType>::N_Warp;
    static constexpr bool TiledMMAPermuteN = false; // N_Repeat % 2 == 0;
};

struct MXMoeFlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct MXGemmTypeConfig_fp4xfp4
{
    using ADataType   = ck_tile::pk_fp4_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

struct MXGemmTypeConfig_fp8xfp4
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

struct A16W4_MoeFlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct A16W4_GemmTypeConfig_bf16xfp4
{
    using ADataType   = ck_tile::bfloat16_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bfloat16_t;
};

struct A16W4_GemmTypeConfig_fp16xfp4
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};
