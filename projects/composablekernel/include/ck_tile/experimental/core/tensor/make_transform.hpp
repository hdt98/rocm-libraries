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
#include "ck_tile/experimental/core/tensor/coordinate_transform.hpp"
#include "ck_tile/experimental/core/tensor/magic_division.hpp"
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"

namespace ck_tile {

/** @brief Create a right-sized array of dimension values from variadic arguments.
 *
 *  Eliminates static_array<index_t, N>{...} boilerplate at call sites for
 *  make_merge, make_unmerge, make_embed, and make_tensor_descriptor.
 *
 *  @param vs  Dimension values (variadic, convertible to index_t)
 *  @return static_array<index_t, sizeof...(vs)>
 *
 *  Example: dims(8, 128, 8) -> static_array<index_t, 3>{8, 128, 8}
 */
template <typename... Ts>
constexpr static_array<index_t, sizeof...(Ts)> dims(Ts... vs)
{
    return {static_cast<index_t>(vs)...};
}

/** @brief Create a DimIds routing array with -1 sentinel padding.
 *
 *  Produces a 64-element DimIds array for slot routing in transform graphs.
 *  Unused slots are set to -1 (sentinel). Called by read() and write().
 *
 *  @param is  Dimension indices (variadic)
 *  @return DimIds{is..., -1, -1, ...}
 */
template <typename... Ts>
constexpr DimIds dim_ids(Ts... is)
{
    static_assert(sizeof...(Ts) <= MAX_TENSOR_DIMS,
                  "dim_ids: too many indices (max MAX_TENSOR_DIMS)");
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
    using Impl = TransformImpl<TransformType::PASS_THROUGH>;

    CoordinateTransform t{};
    t.type         = TransformType::PASS_THROUGH;
    t.ndim_input   = 1;
    t.ndim_output  = 1;
    t.is_bijective = true;

    Impl::Schema d{};
    d.dim_length = dim_length;
    Impl::writeSchema(t, d);

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
    static_assert(N <= MAX_TENSOR_DIMS, "make_merge: too many dims (max MAX_TENSOR_DIMS)");
    static_assert(N >= 2, "make_merge: need at least 2 dims to merge");

    using Impl = TransformImpl<TransformType::MERGE>;

    CoordinateTransform t{};
    t.type         = TransformType::MERGE;
    t.ndim_input   = 1;
    t.ndim_output  = N;
    t.is_bijective = true;

    Impl::Schema d{};
    for(index_t i = 0; i < N; ++i)
        d.component_lengths[i] = component_lengths[i];

    // Compute prefix-product strides
    d.strides[N - 1] = 1;
    for(index_t i = N - 2; i >= 0; --i)
        d.strides[i] = d.strides[i + 1] * component_lengths[i + 1];

    // Pre-compute magic division constants
    for(index_t i = 0; i < N - 1; ++i)
        d.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(d.strides[i]));

    Impl::writeSchema(t, d);
    return t;
}

/// Variadic overload: make_merge(K_DIV8, K_MOD8) instead of make_merge(dims(K_DIV8, K_MOD8))
template <typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) >= 2)>::type>
constexpr CoordinateTransform make_merge(Ts... component_lengths)
{
    return make_merge(
        static_array<index_t, sizeof...(Ts)>{static_cast<index_t>(component_lengths)...});
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
    static_assert(N <= MAX_TENSOR_DIMS, "make_unmerge: too many dims (max MAX_TENSOR_DIMS)");
    static_assert(N >= 2, "make_unmerge: need at least 2 dims to unmerge");

    using Impl = TransformImpl<TransformType::UNMERGE>;

    CoordinateTransform t{};
    t.type         = TransformType::UNMERGE;
    t.ndim_input   = N;
    t.ndim_output  = 1;
    t.is_bijective = true;

    Impl::Schema d{};
    for(index_t i = 0; i < N; ++i)
        d.component_lengths[i] = component_lengths[i];

    // Compute prefix-product strides
    d.strides[N - 1] = 1;
    for(index_t i = N - 2; i >= 0; --i)
        d.strides[i] = d.strides[i + 1] * component_lengths[i + 1];

    // Pre-compute magic division for reversal
    for(index_t i = 0; i < N - 1; ++i)
        d.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(d.strides[i]));

    Impl::writeSchema(t, d);
    return t;
}

/// Variadic overload: make_unmerge(3, 4, 5) instead of make_unmerge(dims(3, 4, 5))
template <typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) >= 2)>::type>
constexpr CoordinateTransform make_unmerge(Ts... component_lengths)
{
    return make_unmerge(
        static_array<index_t, sizeof...(Ts)>{static_cast<index_t>(component_lengths)...});
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
    static_assert(N <= MAX_TENSOR_DIMS, "make_embed: too many dims (max MAX_TENSOR_DIMS)");

    using Impl = TransformImpl<TransformType::EMBED>;

    CoordinateTransform t{};
    t.type        = TransformType::EMBED;
    t.ndim_input  = N;
    t.ndim_output = 1;

    Impl::Schema d{};
    for(index_t i = 0; i < N; ++i)
    {
        d.dim_lengths[i] = dim_lengths[i];
        d.strides[i]     = strides[i];
    }

    // Check bijectivity: stride[k] >= stride[k+1] * length[k+1]
    t.is_bijective = true;
    for(index_t i = 0; i < N - 1; ++i)
    {
        if(strides[i] < strides[i + 1] * dim_lengths[i + 1])
            t.is_bijective = false;
        d.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(strides[i]));
    }

    Impl::writeSchema(t, d);
    return t;
}

/** @brief Create an EMBED transform from a TensorDescriptor.
 *
 *  Extracts lengths and strides from the descriptor to build an Embed
 *  transform. Unlike the templated make_embed(lengths, strides) overload,
 *  this accepts a runtime ndim via the descriptor's constexpr fields.
 *
 *  @param desc  Tensor descriptor with lengths and strides
 *  @return CoordinateTransform with type = EMBED, ndim_input = desc.ndim, ndim_output = 1
 */
constexpr CoordinateTransform make_embed(const TensorDescriptor& desc)
{
    using Impl = TransformImpl<TransformType::EMBED>;

    CoordinateTransform t{};
    t.type        = TransformType::EMBED;
    t.ndim_input  = desc.ndim;
    t.ndim_output = 1;

    Impl::Schema d{};
    for(index_t i = 0; i < desc.ndim; ++i)
    {
        d.dim_lengths[i] = desc.lengths[i];
        d.strides[i]     = desc.strides[i];
    }

    t.is_bijective = true;
    for(index_t i = 0; i < desc.ndim - 1; ++i)
    {
        if(desc.strides[i] < desc.strides[i + 1] * desc.lengths[i + 1])
            t.is_bijective = false;
        d.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(desc.strides[i]));
    }

    Impl::writeSchema(t, d);
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
    using Impl = TransformImpl<TransformType::PAD>;

    CoordinateTransform t{};
    t.type              = TransformType::PAD;
    t.ndim_input        = 1;
    t.ndim_output       = 1;
    t.skip_bounds_check = skip_check;
    t.is_bijective      = true;

    Impl::Schema d{};
    d.dim_length = dim_length;
    d.left_pad   = left_pad;
    d.right_pad  = right_pad;
    Impl::writeSchema(t, d);

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
    using Impl = TransformImpl<TransformType::XOR>;

    CoordinateTransform t{};
    t.type         = TransformType::XOR;
    t.ndim_input   = 2;
    t.ndim_output  = 2;
    t.is_bijective = true;

    Impl::Schema d{};
    d.length_0 = length_0;
    d.length_1 = length_1;
    Impl::writeSchema(t, d);

    return t;
}

// ============================================================================
// Graph interface declarations: inputs() + outputs()
// ============================================================================

struct GraphInputs
{
    DimIds slots{};
    constexpr bool operator==(const GraphInputs&) const = default;
};

struct GraphOutputs
{
    DimIds slots{};
    constexpr bool operator==(const GraphOutputs&) const = default;
};

template <typename... Ts>
constexpr GraphInputs inputs(Ts... ids)
{
    return {dim_ids(ids...)};
}

template <typename... Ts>
constexpr GraphOutputs outputs(Ts... ids)
{
    return {dim_ids(ids...)};
}

constexpr index_t count_valid(const DimIds& ids)
{
    index_t n = 0;
    for(index_t i = 0; i < MAX_TENSOR_DIMS; ++i)
        if(ids[i] >= 0)
            n++;
    return n;
}

// ============================================================================
// Transform binding API: transform() + read() + write()
// ============================================================================

/** @brief Specify which slots this transform reads from during traversal.
 *
 *  During traversal (user → memory), the transform reads values from
 *  these slots. In descriptor-based construction, these are relative
 *  to the new user-facing dims. In global-index construction, these
 *  are direct slot positions.
 *
 *  @param ids  Slot indices (variadic)
 *  @return DimIds array with -1 padding for unused slots
 */
template <typename... Ts>
constexpr DimIds read(Ts... ids)
{
    return dim_ids(ids...);
}

/** @brief Specify which slots this transform writes to during traversal.
 *
 *  During traversal (user → memory), the transform writes results to
 *  these slots. In descriptor-based construction, these are relative
 *  to the existing graph's dims. In global-index construction, these
 *  are direct slot positions.
 *
 *  @param ids  Slot indices (variadic)
 *  @return DimIds array with -1 padding for unused slots
 */
template <typename... Ts>
constexpr DimIds write(Ts... ids)
{
    return dim_ids(ids...);
}

/** @brief Bundles a transform with its read/write dimension routing.
 *
 *  Structural NTTP — pure aggregate with defaulted ==.
 *  Created by the transform() factory. Used by make_transform_graph().
 *
 *  Fields follow traversal data flow:
 *    - xform: the coordinate transform to apply
 *    - read_dims: which slots this transform reads from during traversal
 *    - write_dims: which slots this transform writes to during traversal
 */
struct TransformBinding
{
    CoordinateTransform xform{};
    DimIds read_dims{};
    DimIds write_dims{};

    constexpr bool operator==(const TransformBinding&) const = default;
};

/** @brief Bind a transform to its read and write dimension routing.
 *
 *  Parameter order follows traversal data flow: read (input) → write (output).
 *
 *  @param xform      The coordinate transform
 *  @param read_dims  Slots this transform reads from during traversal
 *  @param write_dims Slots this transform writes to during traversal
 *  @return TransformBinding bundling all three
 *
 *  Example:
 *    transform(make_pass_through(128), read(0), write(1))
 *    transform(make_merge(dims(8, 8)), read(1), write(0, 2))
 */
constexpr TransformBinding transform(CoordinateTransform xform, DimIds read_dims, DimIds write_dims)
{
    return {xform, read_dims, write_dims};
}

void transform_error_desc_read_count_mismatch();
void transform_error_desc_write_count_not_one();

/// Bind a TensorDescriptor as an Embed transform with routing.
/// read() count must equal desc.ndim, write() count must be 1.
constexpr TransformBinding
transform(const TensorDescriptor& desc, DimIds read_dims, DimIds write_dims)
{
    if(count_valid(read_dims) != desc.ndim)
        transform_error_desc_read_count_mismatch();
    if(count_valid(write_dims) != 1)
        transform_error_desc_write_count_not_one();
    return {make_embed(desc), read_dims, write_dims};
}

/** @brief Bundles a sub-graph with its read/write boundary routing.
 *
 *  The sub-graph's transforms are expanded inline into the parent graph
 *  during construction. read() maps to the sub-graph's inputs,
 *  write() maps to the sub-graph's outputs.
 */
struct GraphBinding
{
    detail::TransformGraph graph{};
    DimIds read_dims{};
    DimIds write_dims{};

    constexpr bool operator==(const GraphBinding&) const = default;
};

void transform_error_graph_read_count_mismatch();
void transform_error_graph_write_count_mismatch();

/// Bind a TransformGraph as a sub-graph with boundary routing.
/// read() count must equal graph's input count, write() count must equal graph's output count.
constexpr GraphBinding
transform(const detail::TransformGraph& g, DimIds read_dims, DimIds write_dims)
{
    if(count_valid(read_dims) != g.ndim_input)
        transform_error_graph_read_count_mismatch();
    if(count_valid(write_dims) != g.ndim_output)
        transform_error_graph_write_count_mismatch();
    return {g, read_dims, write_dims};
}

} // namespace ck_tile
