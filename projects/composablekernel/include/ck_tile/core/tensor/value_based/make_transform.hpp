// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file make_transform.hpp
 *  @brief Factory functions for constructing CoordinateTransform instances.
 *
 *  These are plain constexpr functions (no __host__ __device__) because they
 *  run only at compile time during graph construction. The resulting
 *  CoordinateTransform values flow to device code via NTTP.
 *
 *  Every factory:
 *  - Guarantees zero-initialization of unused array slots (for NTTP deduplication)
 *  - Includes static_assert bounds guards
 *  - Returns a fully-constructed CoordinateTransform
 */

#pragma once

#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/value_based/coordinate_transform.hpp"
#include "ck_tile/core/tensor/value_based/magic_division.hpp"

namespace ck_tile {

/** @brief Create a DimIds array with -1 padding for unused slots.
 *
 *  Convenience helper for specifying dimension index arrays in apply_transforms().
 *  Unused slots are set to -1 (sentinel value).
 *
 *  @param is  Dimension indices (variadic)
 *  @return DimIds array with values at front, -1 for remaining slots
 *
 *  Example: dims(0, 2) -> DimIds{0, 2, -1, -1, -1}
 */
template <typename... Ts>
constexpr DimIds dims(Ts... is)
{
    static_assert(sizeof...(Ts) <= MAX_DIMS_PER_TRANSFORM,
                  "dims: too many indices (max MAX_DIMS_PER_TRANSFORM)");
    DimIds result{};
    for(auto& x : result.elems)
    {
        x = -1;
    }
    index_t idx = 0;
    ((result[idx++] = static_cast<index_t>(is)), ...);
    return result;
}

/** @brief Create a PASS_THROUGH transform (identity mapping).
 *  @param dim_length  Length of the dimension
 *  @return CoordinateTransform with type = PASS_THROUGH, ndim_input = 1, ndim_output = 1
 */
constexpr CoordinateTransform make_pass_through(index_t dim_length)
{
    CoordinateTransform t{};
    t.type        = TransformType::PASS_THROUGH;
    t.ndim_input  = 1;
    t.ndim_output = 1;
    t.lengths[0]  = dim_length;
    return t;
}

/** @brief Create a MERGE transform (merge N component dims into 1 user-facing dim).
 *
 *  During graph traversal (user → memory), the merge transform receives
 *  1 merged value from the user side and DECOMPOSES it into N component
 *  values toward memory using magic division.
 *
 *  ndim_input = 1 (user side: the single merged value)
 *  ndim_output = N (memory side: the N components)
 *
 *  This matches v1's merge_v2_magic_division where NDimUp=1, NDimLow=N.
 *
 *  @tparam N  Number of component dimensions (deduced from array size)
 *  @param component_lengths  Sizes of each component dimension
 *  @return CoordinateTransform with type = MERGE, ndim_input = 1, ndim_output = N
 *
 *  Example: make_merge({4, 8}) with input=19 → output = {2, 3} (19 = 2*8 + 3)
 */
template <index_t N>
constexpr CoordinateTransform make_merge(const static_array<index_t, N>& component_lengths)
{
    static_assert(N <= MAX_DIMS_PER_TRANSFORM,
                  "make_merge: too many dims (max MAX_DIMS_PER_TRANSFORM)");
    static_assert(N >= 2, "make_merge: need at least 2 dims to merge");

    CoordinateTransform t{};
    t.type        = TransformType::MERGE;
    t.ndim_input  = 1; // user side: 1 merged value
    t.ndim_output = N; // memory side: N component values

    for(index_t i = 0; i < N; ++i)
    {
        t.lengths[i] = component_lengths[i];
    }

    // Pre-compute magic division constants for decomposition at runtime.
    // Divisor for dimension i = product of lengths[i+1..N-1]
    // E.g., lengths = {4, 8}: divisor[0] = 8, magic_divs[0] = magic_div(8)
    // The last dimension uses modulo (remainder), no magic div needed.
    for(index_t i = 0; i < N - 1; ++i)
    {
        index_t divisor = 1;
        for(index_t j = i + 1; j < N; ++j)
        {
            divisor *= component_lengths[j];
        }
        t.coefficients[i] = divisor; // store divisor for remainder computation
        t.magic_divs[i]   = compute_magic_div(static_cast<uint32_t>(divisor));
    }

    return t;
}

/** @brief Create an UNMERGE transform (split 1 memory dim into N user-facing dims).
 *
 *  During graph traversal (user → memory), the unmerge transform receives
 *  N component values from the user side and COMPOSES them into 1 flat
 *  value toward memory using multiply-accumulate.
 *
 *  ndim_input = N (user side: the N component values)
 *  ndim_output = 1 (memory side: the single flat value)
 *
 *  This matches v1's unmerge where NDimUp=N, NDimLow=1.
 *
 *  @tparam N  Number of component dimensions (deduced from array size)
 *  @param component_lengths  Sizes of each component dimension
 *  @return CoordinateTransform with type = UNMERGE, ndim_input = N, ndim_output = 1
 *
 *  Example: make_unmerge({4, 8}) with input={2, 3} → output = 2*8 + 3 = 19
 */
template <index_t N>
constexpr CoordinateTransform make_unmerge(const static_array<index_t, N>& component_lengths)
{
    static_assert(N <= MAX_DIMS_PER_TRANSFORM,
                  "make_unmerge: too many dims (max MAX_DIMS_PER_TRANSFORM)");
    static_assert(N >= 2, "make_unmerge: need at least 2 dims to unmerge");

    CoordinateTransform t{};
    t.type        = TransformType::UNMERGE;
    t.ndim_input  = N; // user side: N component values
    t.ndim_output = 1; // memory side: 1 flat value

    for(index_t i = 0; i < N; ++i)
    {
        t.lengths[i] = component_lengths[i];
    }

    // Compute strides as reverse exclusive scan (prefix products from right)
    // For lengths = {4, 8}: strides = {8, 1}
    // Used for multiply-accumulate composition in mapIndices.
    t.coefficients[N - 1] = 1;
    for(index_t i = N - 2; i >= 0; --i)
    {
        t.coefficients[i] = t.coefficients[i + 1] * component_lengths[i + 1];
    }

    return t;
}

/** @brief Create an EMBED transform (linear combination with strides).
 *
 *  @tparam N  Number of input dimensions (deduced from array size)
 *  @param dim_lengths  Sizes of each input dimension
 *  @param strides      Coefficients for the linear combination
 *  @return CoordinateTransform with type = EMBED, ndim_input = N, ndim_output = 1
 *
 *  Example: make_embed({128, 8}, {8, 1}) -> output = input[0]*8 + input[1]*1
 */
template <index_t N>
constexpr CoordinateTransform make_embed(const static_array<index_t, N>& dim_lengths,
                                         const static_array<index_t, N>& strides)
{
    static_assert(N <= MAX_DIMS_PER_TRANSFORM,
                  "make_embed: too many dims (max MAX_DIMS_PER_TRANSFORM)");

    CoordinateTransform t{};
    t.type        = TransformType::EMBED;
    t.ndim_input  = N;
    t.ndim_output = 1;

    for(index_t i = 0; i < N; ++i)
    {
        t.lengths[i]      = dim_lengths[i];
        t.coefficients[i] = strides[i];
    }

    return t;
}

/** @brief Create a PAD transform (shifted mapping with bounds).
 *
 *  Covers left-only (right_pad=0), right-only (left_pad=0), and both-sides padding.
 *
 *  @param dim_length  Unpadded dimension length
 *  @param left_pad    Left padding amount (0 for right-only)
 *  @param right_pad   Right padding amount (0 for left-only)
 *  @param skip_check  If true, skip bounds validation (e.g., halo regions)
 *  @return CoordinateTransform with type = PAD, ndim_input = 1, ndim_output = 1
 */
constexpr CoordinateTransform
make_pad(index_t dim_length, index_t left_pad, index_t right_pad, bool skip_check = false)
{
    CoordinateTransform t{};
    t.type              = TransformType::PAD;
    t.ndim_input        = 1;
    t.ndim_output       = 1;
    t.lengths[0]        = dim_length;
    t.coefficients[0]   = left_pad;
    t.coefficients[1]   = right_pad;
    t.skip_bounds_check = skip_check;
    return t;
}

/** @brief Create a right-pad-only transform (convenience wrapper).
 *  @param dim_length  Unpadded dimension length
 *  @param pad_amount  Right padding amount
 *  @return CoordinateTransform with type = PAD, left_pad = 0
 */
constexpr CoordinateTransform make_right_pad(index_t dim_length, index_t pad_amount)
{
    return make_pad(dim_length, 0, pad_amount);
}

/** @brief Create an XOR transform for LDS bank conflict avoidance.
 *  @param length_0  Length of first dimension
 *  @param length_1  Length of second dimension
 *  @return CoordinateTransform with type = XOR, ndim_input = 2, ndim_output = 2
 */
constexpr CoordinateTransform make_xor(index_t length_0, index_t length_1)
{
    CoordinateTransform t{};
    t.type        = TransformType::XOR;
    t.ndim_input  = 2;
    t.ndim_output = 2;
    t.lengths[0]  = length_0;
    t.lengths[1]  = length_1;
    return t;
}

} // namespace ck_tile
