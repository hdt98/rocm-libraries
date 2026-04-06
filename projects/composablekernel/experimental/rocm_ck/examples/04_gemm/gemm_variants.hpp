// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Canonical GEMM variant table — single source of truth for all kernel specs.
// Shared between device (.hip) and host (main.cpp) compilation.
//
// Device code uses gemm_variant_spec("name") to look up a spec by name at
// compile time. Host code iterates gemm_variants[] for registry/dispatch.

#pragma once

#include <rocm_ck/gemm_spec.hpp>

#include <string_view>

namespace rocm_ck {

struct GemmVariant
{
    const char* name;
    GemmSpec spec;
};

consteval GemmVariant
make_variant(const char* name, Signature sig, GemmAlgorithm algo, TargetSet targets)
{
    return {name, makeSpec(sig, algo, targets)};
}

/// Convenience: single GpuTarget wraps to TargetSet::only(target).
consteval GemmVariant
make_variant(const char* name, Signature sig, GemmAlgorithm algo, GpuTarget target)
{
    return {name, makeSpec(sig, algo, TargetSet{target})};
}

inline constexpr GemmVariant gemm_variants[] = {
    make_variant("gemm_fp32",
                 Signature{
                     .dtype = DataType::FP32,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_bf16",
                 Signature{
                     .dtype = DataType::BF16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16_w32",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {32, 32, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16_add",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias", .out = "D"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16_add_relu",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                               ReluOp{.in = "D", .out = "E"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    // --- Layout variants: A×B layout combinations beyond the R×C default ---
    make_variant("gemm_fp16_rr",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16_cr",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                 Tensor{.name = "B", .layout = Layout::Row}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    make_variant("gemm_fp16_cc",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                 Tensor{.name = "B", .layout = Layout::Col}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    // --- Split-K: partition K dimension across blockIdx.z ---
    make_variant("gemm_fp16_splitk",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .k_batch     = 4,
                 },
                 TargetSet::cdna()),
    // --- Pipeline V3: compute-optimized pipeline ---
    make_variant("gemm_fp16_v3",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .pipeline    = Pipeline::V3,
                 },
                 TargetSet::cdna()),
    // --- Multi-D: two D tensors (Add+Add: result = A*B + D0 + D1) ---
    make_variant("gemm_fp16_add_add",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias0", .out = "D"},
                               AddOp{.lhs = "D", .rhs = "bias1", .out = "E"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    // --- Batched GEMM: batch dimension via blockIdx.y (runtime, same spec as unbatched) ---
    make_variant("gemm_fp16_batched",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 TargetSet::cdna()),
    // --- Architecture-adaptive: per-arch tile configs (separate variants) ---
    make_variant("gemm_fp16_gfx90a",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 GpuTarget::gfx90a),
    make_variant("gemm_fp16_gfx942",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {256, 256, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {32, 32, 16},
                 },
                 GpuTarget::gfx942),
    // --- Preshuffle: B matrix pre-rearranged for optimal LDS loads ---
    make_variant("gemm_fp16_preshuffle",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .pipeline    = Pipeline::Preshuffle,
                 },
                 TargetSet::cdna()),
    // --- Memory pipeline: LDS-based with Interwave scheduling ---
    make_variant("gemm_fp16_memory",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile         = {128, 128, 32},
                     .block_waves        = {2, 2, 1},
                     .wave_tile          = {16, 16, 16},
                     .pipeline           = Pipeline::Memory,
                     .pipeline_scheduler = PipelineScheduler::Interwave,
                 },
                 TargetSet::cdna()),
    // --- FP8: asymmetric dtype (fp8 inputs, fp16 output, gfx942+ only) ---
    make_variant("gemm_fp8_fnuz",
                 Signature{
                     .dtype   = DataType::FP8_FNUZ,
                     .tensors = {Tensor{.name = "C", .dtype = DataType::FP16}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {32, 32, 16},
                 },
                 TargetSet::family_gfx94()),
    // --- Pipeline V4: compute double-buffer (ping-pong LDS) ---
    make_variant("gemm_fp16_v4",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .pipeline    = Pipeline::V4,
                 },
                 TargetSet::cdna()),
    // --- Padding: non-aligned M/N dimensions with boundary checks ---
    make_variant("gemm_fp16_padded",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .pad_m       = true,
                     .pad_n       = true,
                 },
                 TargetSet::cdna()),
    // --- INT8 GEMM: int8×int8→int32 with integer accumulation ---
    // V3 pipeline required — V1 does not support int8.
    // gfx942+ only — gfx90a emulates int8 MFMA with float MFMA (wrong bit pattern).
    make_variant(
        "gemm_i8",
        Signature{
            .dtype   = DataType::I8,
            .tensors = {Tensor{.name = "C", .dtype = DataType::I32}},
            .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C", .acc_dtype = DataType::I32}},
        },
        GemmAlgorithm{
            .block_tile  = {128, 128, 64},
            .block_waves = {2, 2, 1},
            .wave_tile   = {32, 32, 16},
            .pipeline    = Pipeline::V3,
        },
        TargetSet::family_gfx94()),
    // --- INT4 block-quantized GEMM: fp8 × int4 with per-group fp8 scales → float ---
    // CK Tile BQuant pipeline requires fp8/bf8 compute type and float output.
    make_variant("gemm_i4_bquant",
                 Signature{
                     .dtype   = DataType::FP8_FNUZ,
                     .tensors = {Tensor{.name     = "B",
                                        .dtype    = DataType::I4,
                                        .layout   = Layout::Row,
                                        .quantize = Quantization{.scale_name  = "scale",
                                                                 .scale_dtype = DataType::FP8_FNUZ,
                                                                 .group_size  = 128}},
                                 Tensor{.name = "C", .dtype = DataType::FP32}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 128},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {32, 32, 16},
                     .pipeline    = Pipeline::V3,
                 },
                 TargetSet::family_gfx94()),
    // --- Direct2D epilogue: no LDS shuffle, direct 2D store ---
    // No D tensor support (fused bias/scale requires CShuffle epilogue).
    make_variant("gemm_fp16_direct2d",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .wave_tile   = {16, 16, 16},
                     .epilogue    = EpilogueStrategy::Direct2D,
                 },
                 TargetSet::cdna()),
    // --- WMMA: gfx1151 (RDNA 3.5) with 16×16×16 wave tiles, wave32 ---
    // CK Tile's WarpGemmDispatcher selects WMMA for __gfx11__/__gfx12__ targets.
    // rocm_ck validates gfx1151 only — other RDNA targets need isValidWaveTile()
    // entries and hardware testing before use. Fixed 16×16×16 tile shape.
    // block_waves = {4,2,1} × wave32 = 256 threads (same workgroup size as CDNA).
    make_variant("gemm_fp16_wmma",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {4, 2, 1},
                     .wave_tile   = {16, 16, 16},
                 },
                 GpuTarget::gfx1151),
};

inline constexpr int gemm_variant_count = sizeof(gemm_variants) / sizeof(gemm_variants[0]);

/// Compile-time variant lookup by name. Typos cause compile errors.
consteval GemmSpec gemm_variant_spec(const char* name)
{
    for(const auto& v : gemm_variants)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown GEMM variant name";
}

} // namespace rocm_ck
