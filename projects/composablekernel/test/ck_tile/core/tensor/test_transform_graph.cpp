// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file test_transform_graph.cpp
 *  @brief Constexpr correctness tests for value-based TensorDescriptor and TransformGraph.
 *
 *  All tests use static_assert — if this file compiles, the tests pass.
 *  No runtime execution needed.
 *
 *  Test cases:
 *  1. TensorDescriptor: packed and strided construction
 *  2. TransformGraph from descriptor (Embed)
 *  3. GEMM LDS chain (Embed + PassThrough + Merge)
 *  4. Padded chain (Embed + Pad + PassThrough)
 *  5. Roundtrip (Unmerge -> Merge)
 *  6. Structural properties and canonicalization
 *  7. Graph reversal (memory offset -> user coordinates)
 */

// IWYU: include only what this test directly uses
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp" // TensorDescriptor,
                                                                  // make_tensor_descriptor
#include "ck_tile/experimental/core/tensor/make_graph.hpp"        // detail::make_transform_graph,
                                                                  // detail::applyTransforms,
                                                                  // detail::isGraphBijective
#include "ck_tile/experimental/core/tensor/make_transform.hpp"    // make_pass_through, make_merge,
                                                                  // make_unmerge, make_right_pad,
                                                                  // dims
#include "ck_tile/experimental/core/tensor/magic_division.hpp"    // computeMagicDiv, doMagicDiv

namespace {

using namespace ck_tile;

// ============================================================================
// Helper: constexpr offset computation for manual verification
// ============================================================================

constexpr index_t
manual_strided_offset(index_t i0, index_t i1, index_t i2, index_t s0, index_t s1, index_t s2)
{
    return i0 * s0 + i1 * s1 + i2 * s2;
}

// ============================================================================
// Test 1: TensorDescriptor construction
// ============================================================================

// Packed 2D descriptor
constexpr auto desc_packed_2d = make_tensor_descriptor(static_array<index_t, 2>{4, 8});
static_assert(desc_packed_2d.ndim == 2);
static_assert(desc_packed_2d.lengths[0] == 4);
static_assert(desc_packed_2d.lengths[1] == 8);
static_assert(desc_packed_2d.strides[0] == 8);
static_assert(desc_packed_2d.strides[1] == 1);
static_assert(desc_packed_2d.element_space_size == 32);

// Strided 3D descriptor (GEMM LDS base)
constexpr index_t K_DIV8 = 8;
constexpr index_t M      = 128;
constexpr index_t K_MOD8 = 8;
constexpr index_t S0     = (M + 1) * 8; // = 1032 (padded stride)
constexpr index_t S1     = 8;
constexpr index_t S2     = 1;

constexpr auto desc_strided_3d = make_tensor_descriptor(static_array<index_t, 3>{K_DIV8, M, K_MOD8},
                                                        static_array<index_t, 3>{S0, S1, S2});
static_assert(desc_strided_3d.ndim == 3);
static_assert(desc_strided_3d.element_space_size == 8248);

// Descriptor equality (NTTP deduplication)
constexpr auto desc_a = make_tensor_descriptor(static_array<index_t, 2>{4, 8});
constexpr auto desc_b = make_tensor_descriptor(static_array<index_t, 2>{4, 8});
static_assert(desc_a == desc_b, "Identical descriptors should be equal");

constexpr auto desc_c = make_tensor_descriptor(static_array<index_t, 2>{8, 4});
static_assert(!(desc_a == desc_c), "Different descriptors should not be equal");

// ============================================================================
// Test 2: TransformGraph from descriptor (exercises Embed)
// ============================================================================

constexpr auto packed_2d = detail::make_transform_graph(desc_packed_2d);
static_assert(packed_2d.ndim_upper == 2);
static_assert(packed_2d.ndim_lower == 1);
static_assert(packed_2d.num_transforms == 1);

static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{0, 7}) == 7);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{1, 0}) == 8);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{3, 7}) == 31);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{2, 5}) == 21);

constexpr auto strided_3d = detail::make_transform_graph(desc_strided_3d);
static_assert(strided_3d.ndim_upper == 3);
static_assert(strided_3d.ndim_lower == 1);

static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{0, 0, 0}) == 0);
static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{1, 0, 0}) == 1032);
static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{0, 1, 0}) == 8);
static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{0, 0, 1}) == 1);
static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{2, 5, 3}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));

// ============================================================================
// Test 3: GEMM LDS descriptor chain (Embed + PassThrough + Merge)
// ============================================================================

constexpr index_t DIM_K_DIV8 = 0;
constexpr index_t DIM_M      = 1;
constexpr index_t DIM_K_MOD8 = 2;

constexpr auto gemm_lds_base = detail::make_transform_graph(desc_strided_3d);

constexpr auto gemm_lds = detail::applyTransforms(
    gemm_lds_base,
    static_array<CoordinateTransform, 2>{make_pass_through(M),
                                         make_merge(static_array<index_t, 2>{K_DIV8, K_MOD8})},
    static_array<DimIds, 2>{dims(DIM_M), dims(DIM_K_DIV8, DIM_K_MOD8)},
    static_array<DimIds, 2>{dims(0), dims(1)});

static_assert(gemm_lds.ndim_upper == 2, "gemm_lds: 2 upper dims (M, K)");
static_assert(gemm_lds.ndim_lower == 1, "gemm_lds: 1 lower dim (offset)");
static_assert(gemm_lds.num_transforms == 3);

static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{1, 0}) == 8);
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 1}) == 1);
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{0, 8}) == 1032);
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{5, 19}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63}) ==
              manual_strided_offset(7, 127, 7, S0, S1, S2));
static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{64, 32}) ==
              manual_strided_offset(4, 64, 0, S0, S1, S2));

// ============================================================================
// Test 4: Padded dimension (exercises Pad transform)
// ============================================================================

constexpr auto desc_pad_base =
    make_tensor_descriptor(static_array<index_t, 2>{10, 20}, static_array<index_t, 2>{20, 1});
constexpr auto padded_base = detail::make_transform_graph(desc_pad_base);

constexpr auto padded = detail::applyTransforms(
    padded_base,
    static_array<CoordinateTransform, 2>{make_right_pad(10, 6), make_pass_through(20)},
    static_array<DimIds, 2>{dims(0), dims(1)},
    static_array<DimIds, 2>{dims(0), dims(1)});

static_assert(padded.ndim_upper == 2);
static_assert(detail::calculateOffset<padded>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<padded>(static_array<index_t, 2>{5, 10}) == 110);

// ============================================================================
// Test 5: Roundtrip — Unmerge then Merge back
// ============================================================================

constexpr auto desc_flat = make_tensor_descriptor(static_array<index_t, 1>{32});
constexpr auto flat_base = detail::make_transform_graph(desc_flat);

constexpr auto unmerged = detail::applyTransforms(
    flat_base,
    static_array<CoordinateTransform, 1>{make_unmerge(static_array<index_t, 2>{4, 8})},
    static_array<DimIds, 1>{dims(0)},
    static_array<DimIds, 1>{dims(0, 1)});

static_assert(unmerged.ndim_upper == 2);

constexpr auto roundtrip = detail::applyTransforms(
    unmerged,
    static_array<CoordinateTransform, 1>{make_merge(static_array<index_t, 2>{4, 8})},
    static_array<DimIds, 1>{dims(0, 1)},
    static_array<DimIds, 1>{dims(0)});

static_assert(roundtrip.ndim_upper == 1);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{0}) == 0);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{1}) == 1);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{19}) == 19);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{31}) == 31);

// ============================================================================
// Test 6: Magic division correctness
// ============================================================================

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

static_assert(detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{5, 19}) == 2107);
static_assert(detail::reverseCalculateOffset<gemm_lds>(2107)[0] == 5, "M recovered");
static_assert(detail::reverseCalculateOffset<gemm_lds>(2107)[1] == 19, "K recovered");

static_assert(detail::reverseCalculateOffset<gemm_lds>(0)[0] == 0);
static_assert(detail::reverseCalculateOffset<gemm_lds>(0)[1] == 0);

constexpr index_t off_127_63 = detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63});
static_assert(detail::reverseCalculateOffset<gemm_lds>(off_127_63)[0] == 127);
static_assert(detail::reverseCalculateOffset<gemm_lds>(off_127_63)[1] == 63);

static_assert(detail::reverseCalculateOffset<packed_2d>(21)[0] == 2, "row=2");
static_assert(detail::reverseCalculateOffset<packed_2d>(21)[1] == 5, "col=5");
static_assert(detail::reverseCalculateOffset<packed_2d>(31)[0] == 3, "row=3");
static_assert(detail::reverseCalculateOffset<packed_2d>(31)[1] == 7, "col=7");

// Bijectivity checks
static_assert(detail::isGraphBijective<gemm_lds>(), "GEMM LDS graph is bijective");
static_assert(detail::isGraphBijective<packed_2d>(), "Packed 2D graph is bijective");
static_assert(detail::isGraphBijective<strided_3d>(), "Strided 3D graph is bijective");

// ============================================================================
// Test 8: TensorDescriptor edge cases
// ============================================================================

// 1D packed descriptor (loop body never executes — boundary case)
constexpr auto desc_1d = make_tensor_descriptor(static_array<index_t, 1>{16});
static_assert(desc_1d.ndim == 1);
static_assert(desc_1d.lengths[0] == 16);
static_assert(desc_1d.strides[0] == 1);
static_assert(desc_1d.element_space_size == 16);
static_assert(desc_1d.lengths[1] == 0, "unused slots must be zero for NTTP dedup");
static_assert(desc_1d.strides[1] == 0);

// 6D packed descriptor (MAX_TENSOR_DIMS boundary)
constexpr auto desc_6d = make_tensor_descriptor(static_array<index_t, 6>{2, 3, 4, 5, 6, 7});
static_assert(desc_6d.ndim == 6);
static_assert(desc_6d.strides[5] == 1);
static_assert(desc_6d.strides[4] == 7);
static_assert(desc_6d.strides[3] == 42);
static_assert(desc_6d.strides[2] == 210);
static_assert(desc_6d.strides[1] == 840);
static_assert(desc_6d.strides[0] == 2520);
static_assert(desc_6d.element_space_size == 2 * 3 * 4 * 5 * 6 * 7);

// Single-element tensor (all lengths = 1)
constexpr auto desc_scalar = make_tensor_descriptor(static_array<index_t, 2>{1, 1});
static_assert(desc_scalar.element_space_size == 1);

// Strided element_space_size manual verification
// lengths={4, 8}, strides={16, 1}: ess = 1 + 3*16 + 7*1 = 56
constexpr auto desc_strided_2d =
    make_tensor_descriptor(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{16, 1});
static_assert(desc_strided_2d.element_space_size == 56);

// ============================================================================
// Test 9: Unmerge — direct offset verification (not through roundtrip)
// ============================================================================

// Unmerge {4, 8}: user provides (row, col), maps to flat = row * 8 + col
static_assert(detail::calculateOffset<unmerged>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<unmerged>(static_array<index_t, 2>{0, 7}) == 7);
static_assert(detail::calculateOffset<unmerged>(static_array<index_t, 2>{1, 0}) == 8);
static_assert(detail::calculateOffset<unmerged>(static_array<index_t, 2>{2, 3}) == 19);
static_assert(detail::calculateOffset<unmerged>(static_array<index_t, 2>{3, 7}) == 31);

// Unmerge reverse: offset -> (row, col)
static_assert(detail::reverseCalculateOffset<unmerged>(0)[0] == 0);
static_assert(detail::reverseCalculateOffset<unmerged>(0)[1] == 0);
static_assert(detail::reverseCalculateOffset<unmerged>(19)[0] == 2);
static_assert(detail::reverseCalculateOffset<unmerged>(19)[1] == 3);
static_assert(detail::reverseCalculateOffset<unmerged>(31)[0] == 3);
static_assert(detail::reverseCalculateOffset<unmerged>(31)[1] == 7);

// ============================================================================
// Test 10: Left and both-sides padding
// ============================================================================

constexpr auto desc_pad_lr_base = make_tensor_descriptor(static_array<index_t, 1>{10});
constexpr auto pad_lr_base      = detail::make_transform_graph(desc_pad_lr_base);

// Pad with left_pad=2, right_pad=4: padded length = 10 + 2 + 4 = 16
constexpr auto padded_lr =
    detail::applyTransforms(pad_lr_base,
                            static_array<CoordinateTransform, 1>{make_pad(10, 2, 4)},
                            static_array<DimIds, 1>{dims(0)},
                            static_array<DimIds, 1>{dims(0)});

// User index 2 maps to base index 0 (2 - left_pad = 0)
static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{2}) == 0);
// User index 5 maps to base index 3 (5 - 2 = 3)
static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{5}) == 3);
// User index 11 maps to base index 9 (11 - 2 = 9, last valid)
static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{11}) == 9);

// Pad reverse: base index -> padded index (add left_pad back)
static_assert(detail::reverseCalculateOffset<padded_lr>(0)[0] == 2,
              "base 0 -> padded 2 (0 + left_pad)");
static_assert(detail::reverseCalculateOffset<padded_lr>(9)[0] == 11,
              "base 9 -> padded 11 (9 + left_pad)");

// ============================================================================
// Test 11: Non-bijective graph
// ============================================================================

// Overlapping strides: stride[0]=1 < stride[1]*length[1] = 1*8 = 8
constexpr auto desc_non_bij =
    make_tensor_descriptor(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{1, 1});
constexpr auto graph_non_bij = detail::make_transform_graph(desc_non_bij);
static_assert(!detail::isGraphBijective<graph_non_bij>(), "Overlapping strides => not bijective");

// Boundary: exactly packed (stride[0]=8 == stride[1]*length[1]) => bijective
constexpr auto desc_exact =
    make_tensor_descriptor(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{8, 1});
constexpr auto graph_exact = detail::make_transform_graph(desc_exact);
static_assert(detail::isGraphBijective<graph_exact>(), "Exactly packed is bijective");

// Gapped strides (stride[0]=10 > 8) => bijective (with gaps)
constexpr auto desc_gapped =
    make_tensor_descriptor(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{10, 1});
constexpr auto graph_gapped = detail::make_transform_graph(desc_gapped);
static_assert(detail::isGraphBijective<graph_gapped>(), "Gapped strides are bijective");

// ============================================================================
// Test 12: upperDimLength — dimension length queries
// ============================================================================

static_assert(detail::upperDimLength<packed_2d>(0) == 4, "packed 2D dim 0 length");
static_assert(detail::upperDimLength<packed_2d>(1) == 8, "packed 2D dim 1 length");
static_assert(detail::upperDimLength<gemm_lds>(0) == 128, "GEMM LDS M dim length");
static_assert(detail::upperDimLength<gemm_lds>(1) == 64, "GEMM LDS K dim length");
static_assert(detail::upperDimLength<unmerged>(0) == 4, "unmerged dim 0 length");
static_assert(detail::upperDimLength<unmerged>(1) == 8, "unmerged dim 1 length");
static_assert(detail::upperDimLength<padded_lr>(0) == 16, "padded dim 0 = 10+2+4 = 16");

// ============================================================================
// Test 13: isValidUpper — bounds checking per transform type
// ============================================================================

// PassThrough: always returns true
constexpr auto pt        = make_pass_through(128);
constexpr index_t pt_v[] = {5};
static_assert(TransformImpl<TransformType::PASS_THROUGH>::isValidUpper(pt, pt_v));

// Embed: checks [0, length) per dimension
constexpr auto emb = make_embed(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{8, 1});
constexpr index_t emb_valid[]   = {3, 7, 0, 0, 0};
constexpr index_t emb_oob_neg[] = {-1, 0, 0, 0, 0};
constexpr index_t emb_oob_hi[]  = {4, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::EMBED>::isValidUpper(emb, emb_valid));
static_assert(!TransformImpl<TransformType::EMBED>::isValidUpper(emb, emb_oob_neg));
static_assert(!TransformImpl<TransformType::EMBED>::isValidUpper(emb, emb_oob_hi));

// Merge: checks [0, product_of_lengths)
constexpr auto mrg            = make_merge(static_array<index_t, 2>{4, 8});
constexpr index_t mrg_valid[] = {31, 0, 0, 0, 0};
constexpr index_t mrg_oob[]   = {32, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::MERGE>::isValidUpper(mrg, mrg_valid));
static_assert(!TransformImpl<TransformType::MERGE>::isValidUpper(mrg, mrg_oob));

// Unmerge: checks [0, length) per component
constexpr auto umrg            = make_unmerge(static_array<index_t, 2>{4, 8});
constexpr index_t umrg_valid[] = {3, 7, 0, 0, 0};
constexpr index_t umrg_oob[]   = {4, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::UNMERGE>::isValidUpper(umrg, umrg_valid));
static_assert(!TransformImpl<TransformType::UNMERGE>::isValidUpper(umrg, umrg_oob));

// Pad: 3-zone check (left pad, valid, right pad)
constexpr auto pd              = make_pad(10, 2, 4);
constexpr index_t pd_left[]    = {1, 0, 0, 0, 0};
constexpr index_t pd_valid[]   = {5, 0, 0, 0, 0};
constexpr index_t pd_right[]   = {12, 0, 0, 0, 0};
constexpr index_t pd_edge_lo[] = {2, 0, 0, 0, 0};
constexpr index_t pd_edge_hi[] = {11, 0, 0, 0, 0};
static_assert(!TransformImpl<TransformType::PAD>::isValidUpper(pd, pd_left), "left pad zone");
static_assert(TransformImpl<TransformType::PAD>::isValidUpper(pd, pd_valid), "valid zone");
static_assert(!TransformImpl<TransformType::PAD>::isValidUpper(pd, pd_right), "right pad zone");
static_assert(TransformImpl<TransformType::PAD>::isValidUpper(pd, pd_edge_lo), "first valid");
static_assert(TransformImpl<TransformType::PAD>::isValidUpper(pd, pd_edge_hi), "last valid");

// Pad with skip_bounds_check: always valid
constexpr auto pd_skip = make_pad(10, 2, 4, true);
static_assert(TransformImpl<TransformType::PAD>::isValidUpper(pd_skip, pd_left),
              "skip_bounds_check bypasses validation");

// ============================================================================
// Test 14: 3-way merge
// ============================================================================

constexpr auto desc_flat_60 = make_tensor_descriptor(static_array<index_t, 1>{60});
constexpr auto flat_60      = detail::make_transform_graph(desc_flat_60);

constexpr auto unmerged_3 = detail::applyTransforms(
    flat_60,
    static_array<CoordinateTransform, 1>{make_unmerge(static_array<index_t, 3>{3, 4, 5})},
    static_array<DimIds, 1>{dims(0)},
    static_array<DimIds, 1>{dims(0, 1, 2)});

// {1, 2, 3} -> 1*20 + 2*5 + 3*1 = 33
static_assert(detail::calculateOffset<unmerged_3>(static_array<index_t, 3>{1, 2, 3}) == 33);
// {2, 3, 4} -> 2*20 + 3*5 + 4*1 = 59
static_assert(detail::calculateOffset<unmerged_3>(static_array<index_t, 3>{2, 3, 4}) == 59);

// Merge back to 1D and verify roundtrip
constexpr auto remerged_3 = detail::applyTransforms(
    unmerged_3,
    static_array<CoordinateTransform, 1>{make_merge(static_array<index_t, 3>{3, 4, 5})},
    static_array<DimIds, 1>{dims(0, 1, 2)},
    static_array<DimIds, 1>{dims(0)});

static_assert(detail::calculateOffset<remerged_3>(static_array<index_t, 1>{33}) == 33);
static_assert(detail::calculateOffset<remerged_3>(static_array<index_t, 1>{59}) == 59);
static_assert(detail::calculateOffset<remerged_3>(static_array<index_t, 1>{0}) == 0);

// ============================================================================
// Test 15: dims() helper — sentinel padding
// ============================================================================

constexpr auto d0 = dims(3);
static_assert(d0[0] == 3);
static_assert(d0[1] == -1);
static_assert(d0[2] == -1);

constexpr auto d2 = dims(0, 2);
static_assert(d2[0] == 0);
static_assert(d2[1] == 2);
static_assert(d2[2] == -1);

// ============================================================================
// Test 16: Graph structural properties
// ============================================================================

static_assert(packed_2d.num_slots == 3, "packed 2D: 1 offset + 2 upper");
static_assert(strided_3d.num_slots == 4, "strided 3D: 1 offset + 3 upper");
static_assert(gemm_lds.num_slots == 6, "GEMM LDS: 4 base + 2 new");
static_assert(packed_2d.upper_slots[0] == 1, "packed 2D upper dim 0 -> slot 1");
static_assert(packed_2d.upper_slots[1] == 2, "packed 2D upper dim 1 -> slot 2");
static_assert(packed_2d.lower_slots[0] == 0, "packed 2D lower dim 0 -> slot 0");

// ============================================================================
// Test 17: upperLength — per transform type
// ============================================================================

static_assert(TransformImpl<TransformType::PASS_THROUGH>::upperLength(pt, 0) == 128);
static_assert(TransformImpl<TransformType::EMBED>::upperLength(emb, 0) == 4);
static_assert(TransformImpl<TransformType::EMBED>::upperLength(emb, 1) == 8);
static_assert(TransformImpl<TransformType::MERGE>::upperLength(mrg, 0) == 32,
              "Merge upper length = product of component lengths");
static_assert(TransformImpl<TransformType::UNMERGE>::upperLength(umrg, 0) == 4);
static_assert(TransformImpl<TransformType::UNMERGE>::upperLength(umrg, 1) == 8);
static_assert(TransformImpl<TransformType::PAD>::upperLength(pd, 0) == 16,
              "Pad upper length = 10 + 2 + 4 = 16");

// ============================================================================
// Test 18: Additional magic division cases
// ============================================================================

static_assert(doMagicDiv(255, computeMagicDiv(16)) == 15, "255 / 16 = 15");
static_assert(doMagicDiv(256, computeMagicDiv(16)) == 16, "256 / 16 = 16");
static_assert(doMagicDiv(1023, computeMagicDiv(32)) == 31, "1023 / 32 = 31");
static_assert(doMagicDiv(1024, computeMagicDiv(32)) == 32, "1024 / 32 = 32");
static_assert(doMagicDiv(8, computeMagicDiv(8)) == 1, "8 / 8 = 1");
static_assert(doMagicDiv(1000000, computeMagicDiv(1000)) == 1000);

// ============================================================================
// Test 19: XOR factory and traversal
// ============================================================================

constexpr auto xor_xform = make_xor(4, 8);
static_assert(xor_xform.type == TransformType::XOR);
static_assert(xor_xform.ndim_upper == 2);
static_assert(xor_xform.ndim_lower == 2);
static_assert(xor_xform.is_bijective == true);

// XOR schema access via readSchema
constexpr auto xor_schema = TransformImpl<TransformType::XOR>::readSchema(xor_xform);
static_assert(xor_schema.length_0 == 4);
static_assert(xor_schema.length_1 == 8);

// XOR traversal: apply to a 2D graph
constexpr auto desc_xor_base =
    make_tensor_descriptor(static_array<index_t, 2>{4, 8}, static_array<index_t, 2>{8, 1});
constexpr auto xor_base = detail::make_transform_graph(desc_xor_base);
constexpr auto xor_graph =
    detail::applyTransforms(xor_base,
                            static_array<CoordinateTransform, 1>{make_xor(4, 8)},
                            static_array<DimIds, 1>{dims(0, 1)},
                            static_array<DimIds, 1>{dims(0, 1)});

// XOR(row=2, col=5): lower[0]=2, lower[1]=5^(2%8)=5^2=7
// offset = 2*8 + 7 = 23
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{2, 5}) == 23);

// XOR(row=0, col=3): lower[0]=0, lower[1]=3^(0%8)=3^0=3
// offset = 0*8 + 3 = 3
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{0, 3}) == 3);

// XOR(row=3, col=0): lower[0]=3, lower[1]=0^(3%8)=0^3=3
// offset = 3*8 + 3 = 27
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{3, 0}) == 27);

// XOR is self-inverse: reverse(offset) -> original coords
static_assert(detail::reverseCalculateOffset<xor_graph>(23)[0] == 2);
static_assert(detail::reverseCalculateOffset<xor_graph>(23)[1] == 5);

// ============================================================================
// Test 20: transform() / upper() / lower() API — GEMM LDS via new syntax
// ============================================================================

// Same GEMM LDS descriptor as Test 3, but using the new binding API
constexpr auto gemm_lds_new =
    make_transform_graph(desc_strided_3d,
                         transform(make_pass_through(M), upper(0), lower(DIM_M)),
                         transform(make_merge(static_array<index_t, 2>{K_DIV8, K_MOD8}),
                                   upper(1),
                                   lower(DIM_K_DIV8, DIM_K_MOD8)));

// Must produce identical offsets to the old API (Test 3)
static_assert(gemm_lds_new.ndim_upper == 2);
static_assert(gemm_lds_new.ndim_lower == 1);
static_assert(gemm_lds_new.num_transforms == 3);

static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{1, 0}) == 8);
static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{0, 1}) == 1);
static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{5, 19}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));
static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{127, 63}) ==
              manual_strided_offset(7, 127, 7, S0, S1, S2));

// Verify it matches the old API exactly
static_assert(detail::calculateOffset<gemm_lds_new>(static_array<index_t, 2>{64, 32}) ==
              detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{64, 32}));

// ============================================================================
// Test 21: transform() / upper() / lower() API — padded graph
// ============================================================================

constexpr auto padded_new =
    make_transform_graph(desc_pad_base,
                         transform(make_right_pad(10, 6), upper(0), lower(0)),
                         transform(make_pass_through(20), upper(1), lower(1)));

static_assert(padded_new.ndim_upper == 2);
static_assert(detail::calculateOffset<padded_new>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<padded_new>(static_array<index_t, 2>{5, 10}) == 110);

// Must match old API
static_assert(detail::calculateOffset<padded_new>(static_array<index_t, 2>{5, 10}) ==
              detail::calculateOffset<padded>(static_array<index_t, 2>{5, 10}));

// ============================================================================
// Test 22: TransformBinding struct
// ============================================================================

constexpr auto binding = transform(make_pass_through(128), upper(0), lower(1));
static_assert(binding.xform.type == TransformType::PASS_THROUGH);
static_assert(binding.upper_dims[0] == 0);
static_assert(binding.upper_dims[1] == -1);
static_assert(binding.lower_dims[0] == 1);
static_assert(binding.lower_dims[1] == -1);

// upper() and lower() are aliases for dims()
static_assert(upper(0, 2) == dims(0, 2));
static_assert(lower(1) == dims(1));

// ============================================================================
// Test 23: MAX_TENSOR_DIMS (5-way merge/unmerge)
// ============================================================================

constexpr auto desc_flat_720 = make_tensor_descriptor(static_array<index_t, 1>{720});
constexpr auto flat_720      = detail::make_transform_graph(desc_flat_720);

// 5-way unmerge: 720 = 2 * 3 * 4 * 5 * 6
constexpr auto unmerged_5 = detail::applyTransforms(
    flat_720,
    static_array<CoordinateTransform, 1>{make_unmerge(static_array<index_t, 5>{2, 3, 4, 5, 6})},
    static_array<DimIds, 1>{dims(0)},
    static_array<DimIds, 1>{dims(0, 1, 2, 3, 4)});

static_assert(unmerged_5.ndim_upper == 5);
// {1, 2, 3, 4, 5} -> 1*360 + 2*120 + 3*30 + 4*6 + 5*1 = 360+240+90+24+5 = 719
static_assert(detail::calculateOffset<unmerged_5>(static_array<index_t, 5>{1, 2, 3, 4, 5}) == 719);
static_assert(detail::calculateOffset<unmerged_5>(static_array<index_t, 5>{0, 0, 0, 0, 0}) == 0);

// 5-way merge back: roundtrip
constexpr auto remerged_5 = detail::applyTransforms(
    unmerged_5,
    static_array<CoordinateTransform, 1>{make_merge(static_array<index_t, 5>{2, 3, 4, 5, 6})},
    static_array<DimIds, 1>{dims(0, 1, 2, 3, 4)},
    static_array<DimIds, 1>{dims(0)});

static_assert(remerged_5.ndim_upper == 1);
static_assert(detail::calculateOffset<remerged_5>(static_array<index_t, 1>{0}) == 0);
static_assert(detail::calculateOffset<remerged_5>(static_array<index_t, 1>{719}) == 719);
static_assert(detail::calculateOffset<remerged_5>(static_array<index_t, 1>{360}) == 360);

// Reverse 5-way
static_assert(detail::reverseCalculateOffset<remerged_5>(719)[0] == 719);

// ============================================================================
// Test 24: MAX_TENSOR_DIMS boundary (64 dimensions)
// ============================================================================

// Create a 64-dim packed descriptor (all lengths = 1 except last = 2)
consteval auto make_64d_descriptor()
{
    static_array<index_t, 64> lengths{};
    for(index_t i = 0; i < 63; ++i)
        lengths[i] = 1;
    lengths[63] = 2;
    return make_tensor_descriptor(lengths);
}

constexpr auto desc_64d = make_64d_descriptor();
static_assert(desc_64d.ndim == 64);
static_assert(desc_64d.element_space_size == 2);
static_assert(desc_64d.lengths[0] == 1);
static_assert(desc_64d.lengths[63] == 2);
static_assert(desc_64d.strides[63] == 1);
static_assert(desc_64d.strides[62] == 2);

// A more practical high-dim test: 10 dimensions
constexpr auto desc_10d =
    make_tensor_descriptor(static_array<index_t, 10>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
static_assert(desc_10d.ndim == 10);
static_assert(desc_10d.element_space_size == 1024);
static_assert(desc_10d.strides[0] == 512);
static_assert(desc_10d.strides[9] == 1);

// ============================================================================
// Test 25: Descriptor with explicit strides at max dims (5D strided)
// ============================================================================

constexpr auto desc_5d_strided = make_tensor_descriptor(
    static_array<index_t, 5>{4, 8, 16, 32, 64},
    static_array<index_t, 5>{16384 * 2, 16384, 1024, 32, 1}); // padded strides

static_assert(desc_5d_strided.ndim == 5);
// element_space_size = 1 + 3*32768 + 7*16384 + 15*1024 + 31*32 + 63*1
//                    = 1 + 98304 + 114688 + 15360 + 992 + 63 = 229408
static_assert(desc_5d_strided.element_space_size == 229408);

// ============================================================================
// Test 26: 64D Embed transform (TransformGraph from 64D descriptor)
// ============================================================================

// Build a transform graph from the 64D descriptor — creates a 64D Embed
constexpr auto graph_64d = detail::make_transform_graph(desc_64d);
static_assert(graph_64d.ndim_upper == 64);
static_assert(graph_64d.ndim_lower == 1);
static_assert(graph_64d.num_transforms == 1);

// All-zero coordinate -> offset 0
consteval index_t test_64d_zero()
{
    static_array<index_t, 64> coord{};
    return detail::calculateOffset<graph_64d>(coord);
}
static_assert(test_64d_zero() == 0);

// Only the last dim set to 1 -> offset 1 (stride[63] = 1)
consteval index_t test_64d_last()
{
    static_array<index_t, 64> coord{};
    coord[63] = 1;
    return detail::calculateOffset<graph_64d>(coord);
}
static_assert(test_64d_last() == 1);

// 10D graph from 10D descriptor — exercises multi-dim Embed
constexpr auto graph_10d = detail::make_transform_graph(desc_10d);
static_assert(graph_10d.ndim_upper == 10);
static_assert(graph_10d.ndim_lower == 1);

// All ones: offset = sum of strides = 512+256+128+64+32+16+8+4+2+1 = 1023
consteval index_t test_10d_all_ones()
{
    static_array<index_t, 10> coord{};
    for(index_t i = 0; i < 10; ++i)
        coord[i] = 1;
    return detail::calculateOffset<graph_10d>(coord);
}
static_assert(test_10d_all_ones() == 1023);

// Reverse 10D: offset 1023 -> all ones
consteval bool test_10d_reverse()
{
    auto result = detail::reverseCalculateOffset<graph_10d>(1023);
    for(index_t i = 0; i < 10; ++i)
        if(result[i] != 1)
            return false;
    return true;
}
static_assert(test_10d_reverse());

} // anonymous namespace

// If this file compiles, all tests pass.
int main() { return 0; }
