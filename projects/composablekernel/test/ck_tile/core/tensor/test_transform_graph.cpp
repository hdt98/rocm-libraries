// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file test_transform_graph.cpp
 *  @brief Constexpr correctness tests for value-based TransformGraph.
 *
 *  All tests use static_assert — if this file compiles, the tests pass.
 *  No runtime execution needed.
 *
 *  Test cases exercise all 5 implemented transform types:
 *  1. Packed 2D (Unmerge)
 *  2. Strided 3D (Embed)
 *  3. GEMM LDS chain (Embed + PassThrough + Merge)
 *  4. Padded chain (Embed + Pad + PassThrough)
 *  5. Roundtrip (Unmerge -> Merge)
 *  6. Structural properties and canonicalization
 *  7. Graph reversal (memory offset -> user coordinates)
 */

// IWYU: include only what this test directly uses
#include "ck_tile/experimental/core/tensor/make_graph.hpp" // make_strided_graph, make_packed_graph,
                                                           // applyTransforms, reverse_graph,
                                                           // isGraphBijective
#include "ck_tile/experimental/core/tensor/make_transform.hpp" // make_pass_through, make_merge,
                                                               // make_unmerge, make_right_pad, dims
#include "ck_tile/experimental/core/tensor/magic_division.hpp" // computeMagicDiv, doMagicDiv

namespace {

using namespace ck_tile;

// ============================================================================
// Helper: constexpr offset computation for manual verification
// ============================================================================

/// Manually compute offset for a 3D strided tensor: sum(idx[i] * stride[i])
constexpr index_t
manual_strided_offset(index_t i0, index_t i1, index_t i2, index_t s0, index_t s1, index_t s2)
{
    return i0 * s0 + i1 * s1 + i2 * s2;
}

// ============================================================================
// Test 1: Packed 2D tensor via make_packed_graph (exercises Embed)
// ============================================================================

constexpr auto packed_2d = make_packed_graph(static_array<index_t, 2>{4, 8});

// Structural properties
static_assert(packed_2d.ndim_upper == 2, "packed_2d: should have 2 input dims");
static_assert(packed_2d.ndim_lower == 1, "packed_2d: should have 1 output dim");
static_assert(packed_2d.element_space_size == 32, "packed_2d: 4*8 = 32 elements");
static_assert(packed_2d.num_transforms == 1, "packed_2d: 1 transform (embed)");

// Offset calculations: packed layout means stride = {8, 1}
// offset = row * 8 + col
static_assert(calculateOffset<packed_2d>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(calculateOffset<packed_2d>(static_array<index_t, 2>{0, 7}) == 7);
static_assert(calculateOffset<packed_2d>(static_array<index_t, 2>{1, 0}) == 8);
static_assert(calculateOffset<packed_2d>(static_array<index_t, 2>{3, 7}) == 31);
static_assert(calculateOffset<packed_2d>(static_array<index_t, 2>{2, 5}) == 21);

// ============================================================================
// Test 2: Strided 3D tensor via make_strided_graph (exercises Embed)
// ============================================================================

// 3D tensor with padding stride: {8, 128, 8} with strides {1032, 8, 1}
// This mimics the GEMM LDS base descriptor layout
constexpr index_t K_DIV8 = 8; // KPerBlock / 8 = 64 / 8
constexpr index_t M      = 128;
constexpr index_t K_MOD8 = 8;
constexpr index_t S0     = (M + 1) * 8; // = 1032 (padded stride)
constexpr index_t S1     = 8;
constexpr index_t S2     = 1;

constexpr auto strided_3d = make_strided_graph(static_array<index_t, 3>{K_DIV8, M, K_MOD8},
                                               static_array<index_t, 3>{S0, S1, S2});

static_assert(strided_3d.ndim_upper == 3);
static_assert(strided_3d.ndim_lower == 1);
// element_space_size = 1 + (8-1)*1032 + (128-1)*8 + (8-1)*1 = 1 + 7224 + 1016 + 7 = 8248
static_assert(strided_3d.element_space_size == 8248);

// Manual offset: i0*1032 + i1*8 + i2*1
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{0, 0, 0}) == 0);
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{1, 0, 0}) == 1032);
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{0, 1, 0}) == 8);
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{0, 0, 1}) == 1);
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{2, 5, 3}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));
static_assert(calculateOffset<strided_3d>(static_array<index_t, 3>{7, 127, 7}) ==
              manual_strided_offset(7, 127, 7, S0, S1, S2));

// ============================================================================
// Test 3: GEMM LDS descriptor chain (Embed + PassThrough + Merge)
//
// This is the validation target from the plan:
//   Step 0: make_strided_graph({K/8, M, 8}, {(M+1)*8, 8, 1})
//   Step 1: applyTransforms with PassThrough(M) on dim 1, Merge({K/8, 8}) on dims {0, 2}
//
// After transformation:
//   Input dim 0 = M (passed through from old dim 1)
//   Input dim 1 = K (merged from old dims 0 and 2: K = k_div8 * 8 + k_mod8)
// ============================================================================

// Named dimension indices for the base graph
constexpr index_t DIM_K_DIV8 = 0;
constexpr index_t DIM_M      = 1;
constexpr index_t DIM_K_MOD8 = 2;

constexpr auto gemm_lds_base = make_strided_graph(static_array<index_t, 3>{K_DIV8, M, K_MOD8},
                                                  static_array<index_t, 3>{S0, S1, S2});

constexpr auto gemm_lds = applyTransforms(
    gemm_lds_base,
    static_array<CoordinateTransform, 2>{make_pass_through(M),
                                         make_merge(static_array<index_t, 2>{K_DIV8, K_MOD8})},
    // input_dims: PassThrough reads old dim 1 (M), Merge reads old dims 0,2 (K/8, 8)
    static_array<DimIds, 2>{dims(DIM_M), dims(DIM_K_DIV8, DIM_K_MOD8)},
    // output_dims: PassThrough produces new dim 0, Merge produces new dim 1
    static_array<DimIds, 2>{dims(0), dims(1)});

static_assert(gemm_lds.ndim_upper == 2, "gemm_lds: 2 input dims (M, K)");
static_assert(gemm_lds.ndim_lower == 1, "gemm_lds: 1 output dim (offset)");
static_assert(gemm_lds.num_transforms == 3, "gemm_lds: 3 transforms (embed + pt + merge)");
static_assert(gemm_lds.element_space_size == 8248, "gemm_lds: same element space as base");

// Offset = f(m, k) where k = k_div8 * 8 + k_mod8
// The chain: user provides (m, k) → PassThrough copies m to slot for dim 1,
// Merge flattens k into (k_div8, k_mod8) feeding dims 0,2 → Embed computes offset

// (m=0, k=0) → k_div8=0, k_mod8=0 → offset = 0*1032 + 0*8 + 0 = 0
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 0}) == 0);

// (m=1, k=0) → offset = 0*1032 + 1*8 + 0 = 8
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{1, 0}) == 8);

// (m=0, k=1) → k_div8=0, k_mod8=1 → offset = 0*1032 + 0*8 + 1 = 1
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 1}) == 1);

// (m=0, k=8) → k_div8=1, k_mod8=0 → offset = 1*1032 + 0*8 + 0 = 1032
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 8}) == 1032);

// (m=5, k=19) → k_div8=2, k_mod8=3 → offset = 2*1032 + 5*8 + 3 = 2107
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{5, 19}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));

// (m=127, k=63) → k_div8=7, k_mod8=7 → offset = 7*1032 + 127*8 + 7 = 8247
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63}) ==
              manual_strided_offset(7, 127, 7, S0, S1, S2));

// (m=64, k=32) → k_div8=4, k_mod8=0 → offset = 4*1032 + 64*8 + 0 = 4640
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{64, 32}) ==
              manual_strided_offset(4, 64, 0, S0, S1, S2));

// ============================================================================
// Test 4: Padded dimension (exercises Pad transform)
//
// Create a 2D strided tensor, then pad one dimension
// ============================================================================

constexpr auto padded_base =
    make_strided_graph(static_array<index_t, 2>{10, 20}, static_array<index_t, 2>{20, 1});

// Pad dim 0 with right_pad=6 (10 → 16), pass through dim 1
constexpr auto padded = applyTransforms(
    padded_base,
    static_array<CoordinateTransform, 2>{make_right_pad(10, 6), make_pass_through(20)},
    static_array<DimIds, 2>{dims(0), dims(1)},
    static_array<DimIds, 2>{dims(0), dims(1)});

static_assert(padded.ndim_upper == 2);
static_assert(padded.element_space_size == padded_base.element_space_size);

// For right-pad, the mapping is identity (no left pad to subtract)
// (i=0, j=0) → offset = 0*20 + 0 = 0
static_assert(calculateOffset<padded>(static_array<index_t, 2>{0, 0}) == 0);
// (i=5, j=10) → offset = 5*20 + 10 = 110
static_assert(calculateOffset<padded>(static_array<index_t, 2>{5, 10}) == 110);

// ============================================================================
// Test 5: Roundtrip — Unmerge then Merge back (exercises both)
//
// Start with packed 1D (32 elements), unmerge into {4, 8}, merge back to 1D.
// The final offset should equal the original flat index.
// ============================================================================

constexpr auto flat_base = make_packed_graph(static_array<index_t, 1>{32});

// Unmerge: 1D (32) → 2D (4, 8)
constexpr auto unmerged = applyTransforms(
    flat_base,
    static_array<CoordinateTransform, 1>{make_unmerge(static_array<index_t, 2>{4, 8})},
    static_array<DimIds, 1>{dims(0)},
    static_array<DimIds, 1>{dims(0, 1)});

static_assert(unmerged.ndim_upper == 2, "unmerged: 2 input dims after unmerge");

// Merge back: 2D (4, 8) → 1D (32)
constexpr auto roundtrip = applyTransforms(
    unmerged,
    static_array<CoordinateTransform, 1>{make_merge(static_array<index_t, 2>{4, 8})},
    static_array<DimIds, 1>{dims(0, 1)},
    static_array<DimIds, 1>{dims(0)});

static_assert(roundtrip.ndim_upper == 1, "roundtrip: back to 1 input dim");

// Roundtrip: offset(k) == k for all k in [0, 32)
static_assert(calculateOffset<roundtrip>(static_array<index_t, 1>{0}) == 0);
static_assert(calculateOffset<roundtrip>(static_array<index_t, 1>{1}) == 1);
static_assert(calculateOffset<roundtrip>(static_array<index_t, 1>{19}) == 19);
static_assert(calculateOffset<roundtrip>(static_array<index_t, 1>{31}) == 31);

// ============================================================================
// Test 6: Structural properties and canonicalization
// ============================================================================

// Two graphs built from different initial values but identical final structure
// should compare equal (NTTP deduplication).
constexpr auto g_a = make_packed_graph(static_array<index_t, 2>{4, 8});
constexpr auto g_b = make_packed_graph(static_array<index_t, 2>{4, 8});
static_assert(g_a == g_b, "Identical graphs should be equal (canonicalization)");

// Different graphs should NOT be equal
constexpr auto g_c = make_packed_graph(static_array<index_t, 2>{8, 4});
static_assert(!(g_a == g_c), "Different graphs should not be equal");

// Magic division correctness: verify computeMagicDiv produces correct quotients
static_assert(doMagicDiv(19, computeMagicDiv(8)) == 2, "19 / 8 = 2");
static_assert(doMagicDiv(31, computeMagicDiv(8)) == 3, "31 / 8 = 3");
static_assert(doMagicDiv(64, computeMagicDiv(8)) == 8, "64 / 8 = 8");
static_assert(doMagicDiv(0, computeMagicDiv(8)) == 0, "0 / 8 = 0");
static_assert(doMagicDiv(7, computeMagicDiv(8)) == 0, "7 / 8 = 0");
static_assert(doMagicDiv(100, computeMagicDiv(13)) == 7, "100 / 13 = 7");
static_assert(doMagicDiv(127, computeMagicDiv(1)) == 127, "127 / 1 = 127");

// ============================================================================
// Test 7: Graph reversal (memory offset -> user coordinates)
// ============================================================================

// Verify roundtrip: (M, K) -> offset -> (M, K) using reverseCalculateOffset
static_assert(calculateOffset<gemm_lds>(static_array<index_t, 2>{5, 19}) == 2107);
static_assert(reverseCalculateOffset<gemm_lds>(2107)[0] == 5, "M recovered");
static_assert(reverseCalculateOffset<gemm_lds>(2107)[1] == 19, "K recovered");

// More roundtrip tests
static_assert(reverseCalculateOffset<gemm_lds>(0)[0] == 0);
static_assert(reverseCalculateOffset<gemm_lds>(0)[1] == 0);

constexpr index_t off_127_63 = calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63});
static_assert(reverseCalculateOffset<gemm_lds>(off_127_63)[0] == 127, "M=127 recovered");
static_assert(reverseCalculateOffset<gemm_lds>(off_127_63)[1] == 63, "K=63 recovered");

constexpr index_t off_64_32 = calculateOffset<gemm_lds>(static_array<index_t, 2>{64, 32});
static_assert(reverseCalculateOffset<gemm_lds>(off_64_32)[0] == 64, "M=64 recovered");
static_assert(reverseCalculateOffset<gemm_lds>(off_64_32)[1] == 32, "K=32 recovered");

// Reverse a packed graph
static_assert(reverseCalculateOffset<packed_2d>(21)[0] == 2, "row=2 from offset 21");
static_assert(reverseCalculateOffset<packed_2d>(21)[1] == 5, "col=5 from offset 21");
static_assert(reverseCalculateOffset<packed_2d>(31)[0] == 3, "row=3 from offset 31");
static_assert(reverseCalculateOffset<packed_2d>(31)[1] == 7, "col=7 from offset 31");

// Bijectivity checks
static_assert(isGraphBijective<gemm_lds>(), "GEMM LDS graph is bijective");
static_assert(isGraphBijective<packed_2d>(), "Packed 2D graph is bijective");
static_assert(isGraphBijective<strided_3d>(), "Strided 3D graph is bijective");

} // anonymous namespace

// If this file compiles, all tests pass.
int main() { return 0; }
