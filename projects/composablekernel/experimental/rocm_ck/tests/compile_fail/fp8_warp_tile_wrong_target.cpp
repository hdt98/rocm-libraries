// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP8 32x32x64 warp tile on gfx942 (gfx950-only tile).
// Expected error: "warp_tile is not a valid instruction shape for this dtype and target"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = make_spec(
    Signature{.dtype = DataType::FP8_FNUZ, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 64}, {2, 2, 1}, {32, 32, 64}},
    GpuTarget::gfx942);
