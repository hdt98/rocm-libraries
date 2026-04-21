// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the ConvHipConv solver (hipconv-backed grouped 3x3 conv).
//
// Restrictions enforced by ConvHipConv::IsApplicable:
//   - 2D convolution only
//   - fp16 only (currently)
//   - architectures recognised by hipconv (gfx950)
//   - the hipconv library must have a valid kernel for the (params, direction) tuple
//
// The cases below are ported from the hipconv benchmark app's `test_pars`
// vector and are known to be applicable for Fwd/Dgrad/Wgrad on gfx950.

#include "unit_conv_solver.hpp"

namespace {

using TestCase = miopen::unit_tests::ConvTestCase;

// Small representative cases (one per channels-per-group family) for smoke
// runs. CPU verification on the larger cases is too slow for smoke.
auto GetConvSmokeTestCases()
{
    constexpr auto datatype = miopenHalf;
    constexpr auto layout   = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64,  4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c:  w=1, single output column
        TestCase{{datatype, layout, {4, 16, 8, 1}}, {datatype, layout, {16,  8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 8c:  w=1, single output column
        TestCase{{datatype, layout, {4, 32, 8, 1}}, {datatype, layout, {32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 16c: w=1, single output column
        TestCase{{datatype, layout, {4, 64, 8, 1}}, {datatype, layout, {64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // 32c: w=1, single output column
        // clang-format on
    };
}

// Stride-1 grouped cases. Supported by all three directions (Fwd/Bwd/Wrw).
// Cases are grouped by channels-per-group; each group exercises:
//   - boundary widths/heights vs the kernel's tile sizes
//   - asymmetric and oversized padding
//   - odd batch counts (n_fold/wave_n out-of-bounds handling)
//   - single- and multi-wave channel tiling
auto GetConvFullStride1Cases()
{
    constexpr auto datatype = miopenHalf;
    constexpr auto layout   = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        // --- Retinanet (ResNeXt backbone) production shapes (stride 1) ---
        TestCase{{datatype, layout, {32,  128, 200, 200}}, {datatype, layout, { 128,   4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 4c
        TestCase{{datatype, layout, {32,  256, 100, 100}}, {datatype, layout, { 256,   8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 8c
        TestCase{{datatype, layout, {32,  512,  50,  50}}, {datatype, layout, { 512,  16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 16c
        TestCase{{datatype, layout, {32, 1024,  25,  25}}, {datatype, layout, {1024,  32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 32c

        // --- 4 channels per group ---
        TestCase{{datatype, layout, {4,  64,  8,  7}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c: 16 grp, w not div WARP_Q
        TestCase{{datatype, layout, {4,  64,  8,  3}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c: w < WARP_Q
        TestCase{{datatype, layout, {4,  64,  1, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c: h < kh
        TestCase{{datatype, layout, {4,  64,  6, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 16}}, // 4c: h == 2*kh
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 16}}, // 4c: pad=0
        TestCase{{datatype, layout, {4, 128,  8,  9}}, {datatype, layout, {128, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 4c: 32 grp multi-wave
        TestCase{{datatype, layout, {5, 128,  8,  9}}, {datatype, layout, {128, 4, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // 4c: odd N

        // --- 8 channels per group ---
        TestCase{{datatype, layout, {1,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: minimal
        TestCase{{datatype, layout, {4,  16, 16, 33}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: w > BLOCK_Q
        TestCase{{datatype, layout, {4,  16,  8,  7}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: odd w
        TestCase{{datatype, layout, {4,  16,  1, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: h < kh
        TestCase{{datatype, layout, {4,  16,  6, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: h == 2*kh
        TestCase{{datatype, layout, {4,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 2}}, // 8c: pad=0
        TestCase{{datatype, layout, {4,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{0, 1}, {1, 1}, {1, 1}, 2}}, // 8c: asym pad
        TestCase{{datatype, layout, {4,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 0}, {1, 1}, {1, 1}, 2}}, // 8c: asym pad
        TestCase{{datatype, layout, {4,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 2}}, // 8c: pad=2
        TestCase{{datatype, layout, {5,  16,  8, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 8c: odd N
        TestCase{{datatype, layout, {4,  64, 16, 64}}, {datatype, layout, {64, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 8}}, // 8c: multi-wave
        TestCase{{datatype, layout, {4,  64,  3,  5}}, {datatype, layout, {64, 8, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 8}}, // 8c: tiny odd HxW

        // --- 16 channels per group ---
        TestCase{{datatype, layout, {32, 512, 64, 64}}, {datatype, layout, {512, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // idealized Retinanet 64x64
        TestCase{{datatype, layout, {128, 512, 64, 64}}, {datatype, layout, {512, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // idealized Retinanet 64x64 N=128
        TestCase{{datatype, layout, {9, 512, 50, 50}}, {datatype, layout, {512, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // batch not div fold
        TestCase{{datatype, layout, {4, 512,  3,  5}}, {datatype, layout, {512, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 32}}, // tiny odd HxW
        TestCase{{datatype, layout, {7,  32, 50, 50}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // small group count
        TestCase{{datatype, layout, {4,  32,  1, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // h < kh
        TestCase{{datatype, layout, {4,  32,  6, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // h == 2*kh
        TestCase{{datatype, layout, {4,  32,  8,  1}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // w=1
        TestCase{{datatype, layout, {4,  32,  8, 17}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1},  2}}, // w > BLOCK_Q
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1},  2}}, // pad=0
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{0, 1}, {1, 1}, {1, 1},  2}}, // asym pad
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 0}, {1, 1}, {1, 1},  2}}, // asym pad
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{2, 2}, {1, 1}, {1, 1},  2}}, // pad=2

        // --- 32 channels per group ---
        TestCase{{datatype, layout, {1,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: minimal
        TestCase{{datatype, layout, {4,  64, 16, 17}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: w > BLOCK_Q
        TestCase{{datatype, layout, {4,  64,  8,  1}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: w=1
        TestCase{{datatype, layout, {4,  64,  1, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: h < kh
        TestCase{{datatype, layout, {4,  64,  6, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: h == 2*kh
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 2}}, // 32c: pad=0
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{0, 1}, {1, 1}, {1, 1}, 2}}, // 32c: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 0}, {1, 1}, {1, 1}, 2}}, // 32c: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{2, 2}, {1, 1}, {1, 1}, 2}}, // 32c: pad=2
        TestCase{{datatype, layout, {5,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: odd N
        TestCase{{datatype, layout, {4, 128, 16, 32}}, {datatype, layout, {128, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 4}}, // 32c: multi-wave
        TestCase{{datatype, layout, {4, 128,  3,  5}}, {datatype, layout, {128, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 4}}, // 32c: tiny odd HxW

        // --- 32c wave_n (N-tiling) ---
        TestCase{{datatype, layout, {2,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: N=2
        TestCase{{datatype, layout, {3,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: odd N=3
        TestCase{{datatype, layout, {3, 128,  8, 16}}, {datatype, layout, {128, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 4}}, // 32c: N=3 multi-wave
        TestCase{{datatype, layout, {3,  64,  3,  5}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {1, 1}, {1, 1}, 2}}, // 32c: N=3 small spatial
        TestCase{{datatype, layout, {3,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{0, 0}, {1, 1}, {1, 1}, 2}}, // 32c: N=3 pad=0
        // clang-format on
    };
}

// Stride-2 grouped cases EXCLUDING 8c. Supported by Fwd and Bwd; Wrw lacks
// stride-2 kernels in hipconv.
auto GetConvFullStride2Cases()
{
    constexpr auto datatype = miopenHalf;
    constexpr auto layout   = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        // --- Retinanet stride-2 shapes ---
        TestCase{{datatype, layout, {32,  512, 100, 100}}, {datatype, layout, { 512,  16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // 16c s2
        TestCase{{datatype, layout, {32, 1024,  50,  50}}, {datatype, layout, {1024,  32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // 32c s2

        // --- 4c stride 2 ---
        TestCase{{datatype, layout, {4, 128,200,200}}, {datatype, layout, {128, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // 4c s2: Retinanet-shape
        TestCase{{datatype, layout, {4,  64,  8,  9}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: minimal
        TestCase{{datatype, layout, {4,  64,  6, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: h == 2*kh
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{0, 0}, {2, 2}, {1, 1}, 16}}, // 4c s2: pad=0
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{0, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 0}, {2, 2}, {1, 1}, 16}}, // 4c s2: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{2, 2}, {2, 2}, {1, 1}, 16}}, // 4c s2: pad=2
        TestCase{{datatype, layout, {5,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: odd N
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: even h pad=1
        TestCase{{datatype, layout, {4, 128, 16, 32}}, {datatype, layout, {128, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // 4c s2: multi-wave
        TestCase{{datatype, layout, {4,  64,  3,  5}}, {datatype, layout, { 64, 4, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 16}}, // 4c s2: tiny odd HxW

        // --- 16c stride 2 ---
        TestCase{{datatype, layout, {4,  32,  8,  9}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2}}, // 16c s2: minimal
        TestCase{{datatype, layout, {4,  32,  6, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2}}, // 16c s2: h == 2*kh
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{0, 0}, {2, 2}, {1, 1},  2}}, // 16c s2: pad=0
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{0, 1}, {2, 2}, {1, 1},  2}}, // 16c s2: asym pad
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 0}, {2, 2}, {1, 1},  2}}, // 16c s2: asym pad
        TestCase{{datatype, layout, {4,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{2, 2}, {2, 2}, {1, 1},  2}}, // 16c s2: pad=2
        TestCase{{datatype, layout, {5,  32,  8, 16}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2}}, // 16c s2: odd N
        TestCase{{datatype, layout, {4, 512, 16, 32}}, {datatype, layout, {512, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // 16c s2: multi-wave
        TestCase{{datatype, layout, {4,  32,  3,  5}}, {datatype, layout, { 32, 16, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1},  2}}, // 16c s2: tiny odd HxW

        // --- 32c stride 2 ---
        TestCase{{datatype, layout, {4,  64,  8,  9}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 32c s2: minimal
        TestCase{{datatype, layout, {4,  64,  6, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 32c s2: h == 2*kh
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{0, 0}, {2, 2}, {1, 1}, 2}}, // 32c s2: pad=0
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{0, 1}, {2, 2}, {1, 1}, 2}}, // 32c s2: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 0}, {2, 2}, {1, 1}, 2}}, // 32c s2: asym pad
        TestCase{{datatype, layout, {4,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{2, 2}, {2, 2}, {1, 1}, 2}}, // 32c s2: pad=2
        TestCase{{datatype, layout, {5,  64,  8, 16}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 32c s2: odd N
        TestCase{{datatype, layout, {4, 128, 16, 32}}, {datatype, layout, {128, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 4}}, // 32c s2: multi-wave
        TestCase{{datatype, layout, {4,  64,  3,  5}}, {datatype, layout, { 64, 32, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 32c s2: tiny odd HxW
        // clang-format on
    };
}

// 8c stride-2 cases. Supported by Fwd only; Bwd/Wrw lack 8c stride-2 kernels.
auto GetConvFullStride2EightCases()
{
    constexpr auto datatype = miopenHalf;
    constexpr auto layout   = miopenTensorNHWC;
    return std::vector<TestCase>{
        // clang-format off
        TestCase{{datatype, layout, {32,  256, 200, 200}}, {datatype, layout, { 256,   8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 32}}, // Retinanet 8c s2
        TestCase{{datatype, layout, {4,  16, 16, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 8c s2: standard
        TestCase{{datatype, layout, {4,  16,  8,  7}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 8c s2: odd output w
        TestCase{{datatype, layout, {4,  16,  3, 32}}, {datatype, layout, {16, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 2}}, // 8c s2: small h
        TestCase{{datatype, layout, {4,  64, 16, 64}}, {datatype, layout, {64, 8, 3, 3}}, datatype, {{1, 1}, {2, 2}, {1, 1}, 8}}, // 8c s2: multi-wave
        // clang-format on
    };
}

// Composes per-direction Full sets:
//   Fwd = stride1 + stride2 + stride2_8c
//   Bwd = stride1 + stride2
//   Wrw = stride1
auto GetConvFullTestCasesFwd()
{
    auto v   = GetConvFullStride1Cases();
    auto s2  = GetConvFullStride2Cases();
    auto s28 = GetConvFullStride2EightCases();
    v.insert(v.end(), s2.begin(), s2.end());
    v.insert(v.end(), s28.begin(), s28.end());
    return v;
}

auto GetConvFullTestCasesBwd()
{
    auto v  = GetConvFullStride1Cases();
    auto s2 = GetConvFullStride2Cases();
    v.insert(v.end(), s2.begin(), s2.end());
    return v;
}

auto GetConvFullTestCasesWrw() { return GetConvFullStride1Cases(); }

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx950);
        p.Tunable(5);
        p.UsesHipconvArchBackend();
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverConvHipConvFwd_FP16 = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvHipConvBwd_FP16 = GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverConvHipConvWrw_FP16 = GPU_UnitTestConvSolverWrw_FP16;

using CPU_UnitTestConvSolverConvHipConvDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;
using CPU_UnitTestConvSolverConvHipConvDevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;
using CPU_UnitTestConvSolverConvHipConvDevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverConvHipConvFwd_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvBwd_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(GPU_UnitTestConvSolverConvHipConvWrw_FP16, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(CPU_UnitTestConvSolverConvHipConvDevApplicabilityFwd_NONE, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(CPU_UnitTestConvSolverConvHipConvDevApplicabilityBwd_NONE, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

TEST_P(CPU_UnitTestConvSolverConvHipConvDevApplicabilityWrw_NONE, ConvHipConv)
{
    this->RunTest(miopen::solver::conv::ConvHipConv{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvHipConvWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvSmokeTestCases())));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvHipConvFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCasesFwd())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvHipConvBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCasesBwd())));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverConvHipConvWrw_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvFullTestCasesWrw())));

// Device applicability tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvHipConvDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases()[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvHipConvDevApplicabilityBwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases()[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvHipConvDevApplicabilityWrw_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvSmokeTestCases()[0])));
