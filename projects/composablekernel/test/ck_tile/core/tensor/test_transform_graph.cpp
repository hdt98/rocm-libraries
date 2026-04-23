// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file test_transform_graph.cpp
 *  @brief Constexpr correctness tests for value-based TensorDescriptor and TransformGraph.
 *
 *  All tests use static_assert — if this file compiles, the tests pass.
 *  No runtime execution needed.
 */

#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/tensor/make_graph.hpp"
#include "ck_tile/experimental/core/tensor/make_transform.hpp"
#include "ck_tile/experimental/core/tensor/magic_division.hpp"

namespace {

using namespace ck_tile;

constexpr index_t
manual_strided_offset(index_t i0, index_t i1, index_t i2, index_t s0, index_t s1, index_t s2)
{
    return i0 * s0 + i1 * s1 + i2 * s2;
}

// ============================================================================
// Test 1: TensorDescriptor construction
// ============================================================================

constexpr auto desc_packed_2d = make_tensor_descriptor(dims(4, 8));
static_assert(desc_packed_2d.ndim == 2);
static_assert(desc_packed_2d.lengths[0] == 4);
static_assert(desc_packed_2d.lengths[1] == 8);
static_assert(desc_packed_2d.strides[0] == 8);
static_assert(desc_packed_2d.strides[1] == 1);
static_assert(desc_packed_2d.element_space_size == 32);

constexpr index_t K_DIV8 = 8;
constexpr index_t M      = 128;
constexpr index_t K_MOD8 = 8;
constexpr index_t S0     = (M + 1) * 8;
constexpr index_t S1     = 8;
constexpr index_t S2     = 1;

constexpr auto desc_strided_3d = make_tensor_descriptor(dims(K_DIV8, M, K_MOD8), dims(S0, S1, S2));
static_assert(desc_strided_3d.ndim == 3);
static_assert(desc_strided_3d.element_space_size == 8248);

constexpr auto desc_a = make_tensor_descriptor(dims(4, 8));
constexpr auto desc_b = make_tensor_descriptor(dims(4, 8));
static_assert(desc_a == desc_b);

constexpr auto desc_c = make_tensor_descriptor(dims(8, 4));
static_assert(!(desc_a == desc_c));

// ============================================================================
// Test 2: TensorDescriptor edge cases
// ============================================================================

constexpr auto desc_1d = make_tensor_descriptor(dims(16));
static_assert(desc_1d.ndim == 1);
static_assert(desc_1d.strides[0] == 1);
static_assert(desc_1d.element_space_size == 16);
static_assert(desc_1d.lengths[1] == 0);
static_assert(desc_1d.strides[1] == 0);

constexpr auto desc_6d = make_tensor_descriptor(dims(2, 3, 4, 5, 6, 7));
static_assert(desc_6d.ndim == 6);
static_assert(desc_6d.element_space_size == 2 * 3 * 4 * 5 * 6 * 7);

constexpr auto desc_scalar = make_tensor_descriptor(dims(1, 1));
static_assert(desc_scalar.element_space_size == 1);

constexpr auto desc_strided_2d = make_tensor_descriptor(dims(4, 8), dims(16, 1));
static_assert(desc_strided_2d.element_space_size == 56);

constexpr auto desc_5d_strided =
    make_tensor_descriptor(dims(4, 8, 16, 32, 64), dims(16384 * 2, 16384, 1024, 32, 1));
static_assert(desc_5d_strided.element_space_size == 229408);

// ============================================================================
// Test 3: Magic division
// ============================================================================

static_assert(doMagicDiv(19, computeMagicDiv(8)) == 2);
static_assert(doMagicDiv(31, computeMagicDiv(8)) == 3);
static_assert(doMagicDiv(64, computeMagicDiv(8)) == 8);
static_assert(doMagicDiv(0, computeMagicDiv(8)) == 0);
static_assert(doMagicDiv(7, computeMagicDiv(8)) == 0);
static_assert(doMagicDiv(100, computeMagicDiv(13)) == 7);
static_assert(doMagicDiv(127, computeMagicDiv(1)) == 127);
static_assert(doMagicDiv(255, computeMagicDiv(16)) == 15);
static_assert(doMagicDiv(1023, computeMagicDiv(32)) == 31);
static_assert(doMagicDiv(1000000, computeMagicDiv(1000)) == 1000);

// ============================================================================
// Test 4: dim_ids() helper — sentinel padding
// ============================================================================

constexpr auto d0 = dim_ids(3);
static_assert(d0[0] == 3);
static_assert(d0[1] == -1);

constexpr auto d2 = dim_ids(0, 2);
static_assert(d2[0] == 0);
static_assert(d2[1] == 2);
static_assert(d2[2] == -1);

// ============================================================================
// Test 5: isValidInput — bounds checking per transform type
// ============================================================================

constexpr auto pt        = make_pass_through(128);
constexpr index_t pt_v[] = {5};
static_assert(TransformImpl<TransformType::PASS_THROUGH>::isValidInput(pt, pt_v));

constexpr auto emb              = make_embed(dims(4, 8), dims(8, 1));
constexpr index_t emb_valid[]   = {3, 7, 0, 0, 0};
constexpr index_t emb_oob_neg[] = {-1, 0, 0, 0, 0};
constexpr index_t emb_oob_hi[]  = {4, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::EMBED>::isValidInput(emb, emb_valid));
static_assert(!TransformImpl<TransformType::EMBED>::isValidInput(emb, emb_oob_neg));
static_assert(!TransformImpl<TransformType::EMBED>::isValidInput(emb, emb_oob_hi));

constexpr auto mrg            = make_merge(4, 8);
constexpr index_t mrg_valid[] = {31, 0, 0, 0, 0};
constexpr index_t mrg_oob[]   = {32, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::MERGE>::isValidInput(mrg, mrg_valid));
static_assert(!TransformImpl<TransformType::MERGE>::isValidInput(mrg, mrg_oob));

constexpr auto umrg            = make_unmerge(4, 8);
constexpr index_t umrg_valid[] = {3, 7, 0, 0, 0};
constexpr index_t umrg_oob[]   = {4, 0, 0, 0, 0};
static_assert(TransformImpl<TransformType::UNMERGE>::isValidInput(umrg, umrg_valid));
static_assert(!TransformImpl<TransformType::UNMERGE>::isValidInput(umrg, umrg_oob));

constexpr auto pd              = make_pad(10, 2, 4);
constexpr index_t pd_left[]    = {1, 0, 0, 0, 0};
constexpr index_t pd_valid[]   = {5, 0, 0, 0, 0};
constexpr index_t pd_right[]   = {12, 0, 0, 0, 0};
constexpr index_t pd_edge_lo[] = {2, 0, 0, 0, 0};
constexpr index_t pd_edge_hi[] = {11, 0, 0, 0, 0};
static_assert(!TransformImpl<TransformType::PAD>::isValidInput(pd, pd_left));
static_assert(TransformImpl<TransformType::PAD>::isValidInput(pd, pd_valid));
static_assert(!TransformImpl<TransformType::PAD>::isValidInput(pd, pd_right));
static_assert(TransformImpl<TransformType::PAD>::isValidInput(pd, pd_edge_lo));
static_assert(TransformImpl<TransformType::PAD>::isValidInput(pd, pd_edge_hi));

constexpr auto pd_skip = make_pad(10, 2, 4, true);
static_assert(TransformImpl<TransformType::PAD>::isValidInput(pd_skip, pd_left));

// ============================================================================
// Test 6: inputLength per transform type
// ============================================================================

static_assert(TransformImpl<TransformType::PASS_THROUGH>::inputLength(pt, 0) == 128);
static_assert(TransformImpl<TransformType::EMBED>::inputLength(emb, 0) == 4);
static_assert(TransformImpl<TransformType::EMBED>::inputLength(emb, 1) == 8);
static_assert(TransformImpl<TransformType::MERGE>::inputLength(mrg, 0) == 32);
static_assert(TransformImpl<TransformType::UNMERGE>::inputLength(umrg, 0) == 4);
static_assert(TransformImpl<TransformType::UNMERGE>::inputLength(umrg, 1) == 8);
static_assert(TransformImpl<TransformType::PAD>::inputLength(pd, 0) == 16);

// ============================================================================
// Test 7: XOR factory
// ============================================================================

constexpr auto xor_xform = make_xor(4, 8);
static_assert(xor_xform.type == TransformType::XOR);
static_assert(xor_xform.ndim_input == 2);
static_assert(xor_xform.ndim_output == 2);
static_assert(xor_xform.is_bijective == true);

constexpr auto xor_schema = TransformImpl<TransformType::XOR>::readSchema(xor_xform);
static_assert(xor_schema.length_0 == 4);
static_assert(xor_schema.length_1 == 8);

// ============================================================================
// Test 8: TransformBinding struct
// ============================================================================

constexpr auto binding = transform(make_pass_through(128), read(0), write(1));
static_assert(binding.xform.type == TransformType::PASS_THROUGH);
static_assert(binding.read_dims[0] == 0);
static_assert(binding.read_dims[1] == -1);
static_assert(binding.write_dims[0] == 1);
static_assert(binding.write_dims[1] == -1);

static_assert(read(0, 2) == dim_ids(0, 2));
static_assert(write(1) == dim_ids(1));

// ============================================================================
// Test 9: make_embed(TensorDescriptor) overload
// ============================================================================

constexpr auto embed_from_desc = make_embed(desc_strided_3d);
static_assert(embed_from_desc.type == TransformType::EMBED);
static_assert(embed_from_desc.ndim_input == 3);
static_assert(embed_from_desc.ndim_output == 1);
static_assert(embed_from_desc.is_bijective == true);

// ============================================================================
// Test 10: make_transform_graph(desc) — single-descriptor convenience
// ============================================================================

constexpr auto packed_2d = make_transform_graph(desc_packed_2d);
static_assert(packed_2d.ndim_input == 2);
static_assert(packed_2d.ndim_output == 1);
static_assert(packed_2d.num_transforms == 1);

static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{3, 7}) == 31);
static_assert(detail::calculateOffset<packed_2d>(static_array<index_t, 2>{2, 5}) == 21);

constexpr auto strided_3d = make_transform_graph(desc_strided_3d);
static_assert(strided_3d.ndim_input == 3);
static_assert(detail::calculateOffset<strided_3d>(static_array<index_t, 3>{2, 5, 3}) ==
              manual_strided_offset(2, 5, 3, S0, S1, S2));

// ============================================================================
// Test 11: make_transform_graph(graph) — identity
// ============================================================================

constexpr auto strided_3d_copy = make_transform_graph(strided_3d);
static_assert(strided_3d_copy == strided_3d);

// ============================================================================
// Test 12: GEMM LDS — one-shot explicit (3 transforms)
// ============================================================================

constexpr index_t SL_OFFSET = 0;
constexpr index_t SL_KDIV8 = 1, SL_M_PHYS = 2, SL_KMOD8 = 3;
constexpr index_t SL_M_USER = 4, SL_K_USER = 5;

constexpr auto gemm_lds = make_transform_graph(
    outputs(SL_OFFSET),
    transform(desc_strided_3d, read(SL_KDIV8, SL_M_PHYS, SL_KMOD8), write(SL_OFFSET)),
    transform(make_pass_through(M), read(SL_M_USER), write(SL_M_PHYS)),
    transform(make_merge(K_DIV8, K_MOD8), read(SL_K_USER), write(SL_KDIV8, SL_KMOD8)),
    inputs(SL_M_USER, SL_K_USER));

static_assert(gemm_lds.ndim_input == 2);
static_assert(gemm_lds.ndim_output == 1);
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
// Test 13: Embed-only one-shot
// ============================================================================

constexpr index_t SL_E_OFF = 0, SL_E_ROW = 1, SL_E_COL = 2;

constexpr auto embed_only = make_transform_graph(
    outputs(SL_E_OFF),
    transform(make_embed(dims(4, 8), dims(8, 1)), read(SL_E_ROW, SL_E_COL), write(SL_E_OFF)),
    inputs(SL_E_ROW, SL_E_COL));

static_assert(embed_only.ndim_input == 2);
static_assert(embed_only.ndim_output == 1);
static_assert(detail::calculateOffset<embed_only>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<embed_only>(static_array<index_t, 2>{3, 7}) == 31);
static_assert(detail::calculateOffset<embed_only>(static_array<index_t, 2>{2, 5}) == 21);

// ============================================================================
// Test 14: Roundtrip — Embed + Unmerge + Merge (3-layer chain)
// ============================================================================

constexpr index_t SL_R_OFF = 0, SL_R_FLAT = 1, SL_R_K0 = 2, SL_R_K1 = 3, SL_R_USER = 4;

constexpr auto roundtrip =
    make_transform_graph(outputs(SL_R_OFF),
                         transform(make_embed(dims(32), dims(1)), read(SL_R_FLAT), write(SL_R_OFF)),
                         transform(make_unmerge(4, 8), read(SL_R_K0, SL_R_K1), write(SL_R_FLAT)),
                         transform(make_merge(4, 8), read(SL_R_USER), write(SL_R_K0, SL_R_K1)),
                         inputs(SL_R_USER));

static_assert(roundtrip.ndim_input == 1);
static_assert(roundtrip.ndim_output == 1);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{0}) == 0);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{19}) == 19);
static_assert(detail::calculateOffset<roundtrip>(static_array<index_t, 1>{31}) == 31);

// ============================================================================
// Test 15: Padded dimension (Embed + Pad + PassThrough)
// ============================================================================

constexpr index_t SL_P_OFF = 0, SL_P_ROW = 1, SL_P_COL = 2;
constexpr index_t SL_P_PADROW = 3, SL_P_UCOL = 4;

constexpr auto padded = make_transform_graph(
    outputs(SL_P_OFF),
    transform(make_embed(dims(10, 20), dims(20, 1)), read(SL_P_ROW, SL_P_COL), write(SL_P_OFF)),
    transform(make_right_pad(10, 6), read(SL_P_PADROW), write(SL_P_ROW)),
    transform(make_pass_through(20), read(SL_P_UCOL), write(SL_P_COL)),
    inputs(SL_P_PADROW, SL_P_UCOL));

static_assert(padded.ndim_input == 2);
static_assert(detail::calculateOffset<padded>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<padded>(static_array<index_t, 2>{5, 10}) == 110);

// ============================================================================
// Test 16: Left/both-sides padding
// ============================================================================

constexpr index_t SL_LP_OFF = 0, SL_LP_BASE = 1, SL_LP_PAD = 2;

constexpr auto padded_lr = make_transform_graph(
    outputs(SL_LP_OFF),
    transform(make_embed(dims(10), dims(1)), read(SL_LP_BASE), write(SL_LP_OFF)),
    transform(make_pad(10, 2, 4), read(SL_LP_PAD), write(SL_LP_BASE)),
    inputs(SL_LP_PAD));

static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{2}) == 0);
static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{5}) == 3);
static_assert(detail::calculateOffset<padded_lr>(static_array<index_t, 1>{11}) == 9);

static_assert(detail::reverseCalculateOffset<padded_lr>(0)[0] == 2);
static_assert(detail::reverseCalculateOffset<padded_lr>(9)[0] == 11);

// ============================================================================
// Test 17: XOR graph (Embed + XOR)
// ============================================================================

constexpr index_t SL_X_OFF = 0, SL_X_PROW = 1, SL_X_PCOL = 2;
constexpr index_t SL_X_UROW = 3, SL_X_UCOL = 4;

constexpr auto xor_graph = make_transform_graph(
    outputs(SL_X_OFF),
    transform(make_embed(dims(4, 8), dims(8, 1)), read(SL_X_PROW, SL_X_PCOL), write(SL_X_OFF)),
    transform(make_xor(4, 8), read(SL_X_UROW, SL_X_UCOL), write(SL_X_PROW, SL_X_PCOL)),
    inputs(SL_X_UROW, SL_X_UCOL));

static_assert(xor_graph.ndim_input == 2);
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{2, 5}) == 23);
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{0, 3}) == 3);
static_assert(detail::calculateOffset<xor_graph>(static_array<index_t, 2>{3, 0}) == 27);

// ============================================================================
// Test 18: 3-way merge/unmerge
// ============================================================================

constexpr index_t SL_3_OFF = 0, SL_3_FLAT = 1;
constexpr index_t SL_3_A = 2, SL_3_B = 3, SL_3_C = 4;
constexpr index_t SL_3_USER = 5;

constexpr auto unmerged_3 = make_transform_graph(
    outputs(SL_3_OFF),
    transform(make_embed(dims(60), dims(1)), read(SL_3_FLAT), write(SL_3_OFF)),
    transform(make_unmerge(3, 4, 5), read(SL_3_A, SL_3_B, SL_3_C), write(SL_3_FLAT)),
    transform(make_merge(3, 4, 5), read(SL_3_USER), write(SL_3_A, SL_3_B, SL_3_C)),
    inputs(SL_3_USER));

static_assert(detail::calculateOffset<unmerged_3>(static_array<index_t, 1>{33}) == 33);
static_assert(detail::calculateOffset<unmerged_3>(static_array<index_t, 1>{59}) == 59);
static_assert(detail::calculateOffset<unmerged_3>(static_array<index_t, 1>{0}) == 0);

// ============================================================================
// Test 19: 5-way merge/unmerge
// ============================================================================

constexpr index_t SL_5_OFF = 0, SL_5_FLAT = 1;
constexpr index_t SL_5_A = 2, SL_5_B = 3, SL_5_C = 4, SL_5_D = 5, SL_5_E = 6;
constexpr index_t SL_5_USER = 7;

constexpr auto roundtrip_5 = make_transform_graph(
    outputs(SL_5_OFF),
    transform(make_embed(dims(720), dims(1)), read(SL_5_FLAT), write(SL_5_OFF)),
    transform(make_unmerge(2, 3, 4, 5, 6),
              read(SL_5_A, SL_5_B, SL_5_C, SL_5_D, SL_5_E),
              write(SL_5_FLAT)),
    transform(
        make_merge(2, 3, 4, 5, 6), read(SL_5_USER), write(SL_5_A, SL_5_B, SL_5_C, SL_5_D, SL_5_E)),
    inputs(SL_5_USER));

static_assert(roundtrip_5.ndim_input == 1);
static_assert(detail::calculateOffset<roundtrip_5>(static_array<index_t, 1>{0}) == 0);
static_assert(detail::calculateOffset<roundtrip_5>(static_array<index_t, 1>{719}) == 719);
static_assert(detail::calculateOffset<roundtrip_5>(static_array<index_t, 1>{360}) == 360);

// ============================================================================
// Test 20: Graph reversal
// ============================================================================

static_assert(detail::isGraphBijective<gemm_lds>());
static_assert(detail::reverseCalculateOffset<gemm_lds>(2107)[0] == 5);
static_assert(detail::reverseCalculateOffset<gemm_lds>(2107)[1] == 19);

constexpr index_t off_127_63 = detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63});
static_assert(detail::reverseCalculateOffset<gemm_lds>(off_127_63)[0] == 127);
static_assert(detail::reverseCalculateOffset<gemm_lds>(off_127_63)[1] == 63);

static_assert(detail::isGraphBijective<embed_only>());
static_assert(detail::reverseCalculateOffset<embed_only>(21)[0] == 2);
static_assert(detail::reverseCalculateOffset<embed_only>(21)[1] == 5);

static_assert(detail::isGraphBijective<xor_graph>());
static_assert(detail::reverseCalculateOffset<xor_graph>(23)[0] == 2);
static_assert(detail::reverseCalculateOffset<xor_graph>(23)[1] == 5);

// ============================================================================
// Test 21: Non-bijective graph
// ============================================================================

constexpr auto desc_non_bij  = make_tensor_descriptor(dims(4, 8), dims(1, 1));
constexpr auto graph_non_bij = make_transform_graph(desc_non_bij);
static_assert(!detail::isGraphBijective<graph_non_bij>());

constexpr auto desc_exact  = make_tensor_descriptor(dims(4, 8), dims(8, 1));
constexpr auto graph_exact = make_transform_graph(desc_exact);
static_assert(detail::isGraphBijective<graph_exact>());

constexpr auto desc_gapped  = make_tensor_descriptor(dims(4, 8), dims(10, 1));
constexpr auto graph_gapped = make_transform_graph(desc_gapped);
static_assert(detail::isGraphBijective<graph_gapped>());

// ============================================================================
// Test 22: inputDimLength queries
// ============================================================================

static_assert(detail::inputDimLength<gemm_lds>(0) == 128, "M dim length");
static_assert(detail::inputDimLength<gemm_lds>(1) == 64, "K dim length");
static_assert(detail::inputDimLength<embed_only>(0) == 4);
static_assert(detail::inputDimLength<embed_only>(1) == 8);
static_assert(detail::inputDimLength<padded_lr>(0) == 16);

// ============================================================================
// Test 23: 64D descriptor and graph
// ============================================================================

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

constexpr auto graph_64d = make_transform_graph(desc_64d);
static_assert(graph_64d.ndim_input == 64);
static_assert(graph_64d.ndim_output == 1);

consteval index_t test_64d_last()
{
    static_array<index_t, 64> coord{};
    coord[63] = 1;
    return detail::calculateOffset<graph_64d>(coord);
}
static_assert(test_64d_last() == 1);

// 10D graph
constexpr auto desc_10d  = make_tensor_descriptor(dims(2, 2, 2, 2, 2, 2, 2, 2, 2, 2));
constexpr auto graph_10d = make_transform_graph(desc_10d);
static_assert(graph_10d.ndim_input == 10);

consteval index_t test_10d_all_ones()
{
    static_array<index_t, 10> coord{};
    for(index_t i = 0; i < 10; ++i)
        coord[i] = 1;
    return detail::calculateOffset<graph_10d>(coord);
}
static_assert(test_10d_all_ones() == 1023);

consteval bool test_10d_reverse()
{
    auto result = detail::reverseCalculateOffset<graph_10d>(1023);
    for(index_t i = 0; i < 10; ++i)
        if(result[i] != 1)
            return false;
    return true;
}
static_assert(test_10d_reverse());

// ============================================================================
// Test 24: transform(TensorDescriptor) — syntactic sugar
// ============================================================================

constexpr auto gemm_lds_sugar = make_transform_graph(
    outputs(SL_OFFSET),
    transform(desc_strided_3d, read(SL_KDIV8, SL_M_PHYS, SL_KMOD8), write(SL_OFFSET)),
    transform(make_pass_through(M), read(SL_M_USER), write(SL_M_PHYS)),
    transform(make_merge(K_DIV8, K_MOD8), read(SL_K_USER), write(SL_KDIV8, SL_KMOD8)),
    inputs(SL_M_USER, SL_K_USER));

static_assert(detail::calculateOffset<gemm_lds_sugar>(static_array<index_t, 2>{5, 19}) ==
              detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{5, 19}));

// ============================================================================
// Test 25: transform(TransformGraph) — sub-graph embedding
// ============================================================================

constexpr auto sub_upper = make_transform_graph(
    outputs(SL_KDIV8, SL_M_PHYS, SL_KMOD8),
    transform(make_pass_through(M), read(SL_M_USER), write(SL_M_PHYS)),
    transform(make_merge(K_DIV8, K_MOD8), read(SL_K_USER), write(SL_KDIV8, SL_KMOD8)),
    inputs(SL_M_USER, SL_K_USER));

constexpr index_t P_OFF = 10, P_KD = 11, P_MP = 12, P_KM = 13;
constexpr index_t P_M = 14, P_K = 15;

constexpr auto composed =
    make_transform_graph(outputs(P_OFF),
                         transform(desc_strided_3d, read(P_KD, P_MP, P_KM), write(P_OFF)),
                         transform(sub_upper, read(P_M, P_K), write(P_KD, P_MP, P_KM)),
                         inputs(P_M, P_K));

static_assert(composed.ndim_input == 2);
static_assert(composed.ndim_output == 1);

static_assert(detail::calculateOffset<composed>(static_array<index_t, 2>{5, 19}) == 2107);
static_assert(detail::calculateOffset<composed>(static_array<index_t, 2>{0, 0}) == 0);
static_assert(detail::calculateOffset<composed>(static_array<index_t, 2>{127, 63}) ==
              detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{127, 63}));
static_assert(detail::calculateOffset<composed>(static_array<index_t, 2>{64, 32}) ==
              detail::calculateOffset<gemm_lds>(static_array<index_t, 2>{64, 32}));

} // anonymous namespace

// If this file compiles, all tests pass.
int main() { return 0; }
