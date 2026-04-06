// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <tuple>
#include <type_traits>
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "test_gemm_streamk_util.hpp"

using F8   = ck_tile::fp8_t;
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;
using F32  = float;

// Layouts
using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// Persistence
using Persistent    = std::true_type;
using NonPersistent = std::false_type;

// Pipelines
using Mem    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

// Reduction Strategies
using Atomic = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Atomic>;
using Linear = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Linear>;
using Tree   = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Tree>;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

using PassThrough       = ck_tile::element_wise::PassThrough;
using AddScale          = ck_tile::element_wise::AddScale;
using ElementWiseAddAdd = ck_tile::element_wise::MultiDAdd;
using MultiplyMultiply  = ck_tile::element_wise::MultiDMultiply;

// clang-format off

// ========================== CompV3 Pipeline ==========================

// Atomics
using KernelTypesStreamKFp16PersistentAtomicCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy  A1Layout  A1DataType  B1Layout  B1DataType  D0Layout  D0DataType  D1Layout  D1DataType  AElementWise  BElementWise  CDElementWise

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Col,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        AddScale,     AddScale,     ElementWiseAddAdd>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        AddScale,     AddScale,     MultiplyMultiply>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        AddScale,     AddScale,     PassThrough>
>;

using KernelTypesStreamKBf16PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      BF16,       Col,      BF16,       Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf8PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      BF8,        Col,      BF8,        Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp8PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      F8,         Row,      F8,         Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      F8,         Row,      F8,         Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      F8,         Row,      F8,         Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic,            Row,      F8,         Row,      F8,         Row,      BF16,       Row,      BF16,       PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp16NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf16NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Row,      BF16,        Col,      BF16,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf8NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Row,      BF8,        Col,      BF8,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp8NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Row,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Row,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Col,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic,            Col,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

// Linear
using KernelTypesStreamKFp16PersistentLinearCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I16,       I16,       I16,     Persistent,   CompV3,   Linear,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear,            Col,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        AddScale,     AddScale,     ElementWiseAddAdd>
>;

// Tree
using KernelTypesStreamKFp16PersistentTreeCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I16,       I16,       I16,     Persistent,   CompV3,   Tree,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree,            Col,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf16PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Row,      BF16,        Col,      BF16,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf8PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Row,      BF8,        Col,      BF8,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp8PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Row,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Row,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Col,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree,            Col,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp16NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree,            Col,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf16NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Row,      BF16,        Col,      BF16,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKBf8NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Row,      BF8,        Col,      BF8,        Row,      BF16,        Row,      BF16,        PassThrough,  PassThrough,  PassThrough>
>;

using KernelTypesStreamKFp8NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Row,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Row,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Col,      F8,        Col,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree,            Col,      F8,        Row,      F8,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;

// ============================= Other Pipelines =============================

using KernelTypesStreamKPipelines = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   Mem,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   Mem,   Tree,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   Mem,   Linear,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV4,   Atomic,            Row,      F16,        Row,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV4,   Tree,            Row,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV4,   Linear,            Col,      F16,        Col,      F16,        Row,      F16,        Row,      F16,        PassThrough,  PassThrough,  PassThrough>
>;
// clang-format on
