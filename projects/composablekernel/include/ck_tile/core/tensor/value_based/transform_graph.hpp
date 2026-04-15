// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transform_graph.hpp
 *  @brief Value-based coordinate transform graph and NTTP-based free functions.
 *
 *  A TransformGraph is a DAG of CoordinateTransform nodes that maps an
 *  N-dimensional input coordinate to an M-dimensional output coordinate.
 *
 *  The graph is used as an NTTP: template<TransformGraph G> to guarantee
 *  compile-time constant folding of all transform parameters and routing.
 *
 *  The graph owns:
 *  - Transform nodes (WHAT mappings exist)
 *  - Internal routing (which working-array slots each transform reads/writes)
 *  - Input/output endpoint slots
 *  - Metadata (element_space_size, vectorization hints)
 *
 *  The graph knows NOTHING about individual transform algorithms —
 *  it delegates to TransformImpl<Type> via static dispatch.
 *
 *  Internal routing uses a working array of index slots. Transforms fan in/out
 *  through this array. The user never sees slot assignments; they are computed
 *  automatically by factory functions (make_strided_graph, apply_transforms).
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/value_based/coordinate_transform.hpp"
#include "ck_tile/core/tensor/value_based/transform_impl.hpp"

namespace ck_tile {

inline constexpr index_t MAX_TRANSFORMS = 5;  ///< Max transforms per graph
inline constexpr index_t MAX_SLOTS      = 16; ///< Max working array slots
inline constexpr index_t MAX_IO_DIMS    = 6;  ///< Max input or output dimensions

/** @brief A directed acyclic graph of coordinate transforms.
 *
 *  General coordinate mapping: N input dimensions -> M output dimensions.
 *  Used as an NTTP: template<TransformGraph G> to guarantee compile-time
 *  constant folding of all transform parameters.
 *
 *  Common use case: tensor layout descriptor where ndim_output=1 (memory offset).
 *  The graph is general and can map between any two coordinate spaces.
 *
 *  Internally routes indices through a working array of slots. The user
 *  never sees slot assignments -- they are computed by factory functions.
 *
 *  Construction:
 *    constexpr auto g = make_strided_graph(lengths, strides);
 *    constexpr auto g2 = apply_transforms(g, transforms, input_dims, output_dims);
 *
 *  Usage (as NTTP for guaranteed constant folding):
 *    constexpr auto g = ...;
 *    auto offset = calculate_offset<g>(static_array<index_t, 2>{m, k});
 *
 *  @note Internal routing fields (t_input_slots, t_output_slots, input_slots,
 *        output_slots, num_slots) must be public for structural NTTP requirements.
 *        They are assigned by factory functions. Users should not set them directly.
 */
struct TransformGraph
{
    // --- Nodes (transforms) ---
    static_array<CoordinateTransform, MAX_TRANSFORMS> transforms{};
    index_t num_transforms = 0;

    // --- Internal routing (assigned by factory functions, not user-facing) ---
    // Per-transform: which working-array slots each transform reads from / writes to.
    static_array<DimIds, MAX_TRANSFORMS> t_input_slots{};
    static_array<DimIds, MAX_TRANSFORMS> t_output_slots{};
    index_t num_slots = 0;

    // --- Input / output dimensions ---
    index_t ndim_input  = 0;                           ///< Number of input coordinate dimensions
    index_t ndim_output = 0;                           ///< Number of output coordinate dimensions
    static_array<index_t, MAX_IO_DIMS> input_slots{};  ///< Which slots receive input coords
    static_array<index_t, MAX_IO_DIMS> output_slots{}; ///< Which slots hold output coords

    // --- Metadata (tensor-layout specific, zero for general mappings) ---
    index_t element_space_size = 0; ///< Total elements in underlying buffer (incl. padding)
    // TODO: move guaranteed_vector_* to a TensorLayout wrapper when the graph
    // is used for non-tensor-layout mappings (tile distributions, SFCs, etc.)
    static_array<index_t, MAX_SLOTS> guaranteed_vector_lengths{};
    static_array<index_t, MAX_SLOTS> guaranteed_vector_strides{};

    constexpr bool operator==(const TransformGraph&) const = default;
};

static_assert(sizeof(TransformGraph) < 2048,
              "TransformGraph NTTP size canary — consider capacity templating if this fires");

// ============================================================================
// Free functions templated on TransformGraph NTTP
// ============================================================================

/** @brief Apply a single transform (called via fold expression, not directly by user).
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam I  Fold index (0 = last transform applied, num_transforms-1 = first)
 *  @param slots  Working array of index values
 */
template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void apply_single_transform(static_array<index_t, G.num_slots>& slots)
{
    constexpr index_t t      = G.num_transforms - 1 - I; // reverse order
    constexpr auto transform = G.transforms[t];

    // Gather inputs from working array
    static_array<index_t, MAX_DIMS_PER_TRANSFORM> t_in{};
    for(index_t d = 0; d < transform.ndim_input; ++d)
    {
        t_in[d] = slots[G.t_input_slots[t][d]];
    }

    // Static dispatch — TransformImpl<Type> selected at compile time.
    // No switch, no dead branches, no runtime dispatch.
    static_array<index_t, MAX_DIMS_PER_TRANSFORM> t_out{};
    TransformImpl<transform.type>::mapIndices(transform, t_out.elems, t_in.elems);

    // Scatter outputs back to working array
    for(index_t d = 0; d < transform.ndim_output; ++d)
    {
        slots[G.t_output_slots[t][d]] = t_out[d];
    }
}

/** @brief Apply all transforms via fold expression (internal dispatch).
 *
 *  @tparam G   The transform graph (NTTP)
 *  @tparam Is  Index sequence 0..num_transforms-1
 *  @param slots  Working array of index values
 */
template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void apply_all_transforms(static_array<index_t, G.num_slots>& slots,
                                                        sequence<Is...>)
{
    (apply_single_transform<G, Is>(slots), ...);
}

/** @brief Map an input coordinate to an output coordinate.
 *
 *  Core operation of the transform graph. Traverses all transforms from
 *  input side to output side using a fold expression for guaranteed
 *  compile-time dispatch.
 *
 *  @tparam G       The transform graph (NTTP — all values are compile-time constants)
 *  @param[out] output  Output coordinate buffer (size >= G.ndim_output)
 *  @param[in]  input   Input coordinate buffer (size >= G.ndim_input)
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void map(index_t* output, const index_t* input)
{
    static_array<index_t, G.num_slots> slots{};

    // Place input coordinate into assigned slots
    for(index_t i = 0; i < G.ndim_input; ++i)
    {
        slots[G.input_slots[i]] = input[i];
    }

    // Apply each transform in reverse using fold expression.
    // Guarantees compile-time dispatch — no runtime loop unrolling needed.
    apply_all_transforms<G>(slots, make_index_sequence<G.num_transforms>{});

    // Extract output coordinate from assigned slots
    for(index_t i = 0; i < G.ndim_output; ++i)
    {
        output[i] = slots[G.output_slots[i]];
    }
}

/** @brief Convenience: compute memory offset when ndim_output == 1.
 *
 *  Type-safe overload that takes a static_array and validates dimensions
 *  at compile time.
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam N  Number of input dimensions (deduced from array size)
 *  @param input  Input coordinate
 *  @return The single output value (memory offset)
 */
template <TransformGraph G, index_t N>
CK_TILE_HOST_DEVICE constexpr index_t calculate_offset(const static_array<index_t, N>& input)
{
    static_assert(N == G.ndim_input,
                  "calculate_offset: input coordinate size must match graph's ndim_input");
    static_assert(G.ndim_output == 1,
                  "calculate_offset: requires ndim_output == 1 (use map() for N->M)");
    index_t output;
    map<G>(&output, input.elems);
    return output;
}

namespace detail {

/// Check a single transform (compile-time index) for a matching slot.
template <TransformGraph G, index_t T>
CK_TILE_HOST_DEVICE constexpr index_t try_input_length_at(index_t slot)
{
    constexpr auto transform = G.transforms[T];
    for(index_t d = 0; d < transform.ndim_input; ++d)
    {
        if(G.t_input_slots[T][d] == slot)
        {
            return TransformImpl<transform.type>::input_length(transform, d);
        }
    }
    return -1; // not found at this transform
}

/// Fold over all transforms to find the one matching a slot.
template <TransformGraph G, index_t... Ts>
CK_TILE_HOST_DEVICE constexpr index_t input_dim_length_dispatch(index_t slot, sequence<Ts...>)
{
    index_t result = -1;
    // Check transforms in reverse order; keep first valid result
    auto check = [&](index_t candidate) {
        if(result == -1 && candidate != -1)
        {
            result = candidate;
        }
    };
    (check(try_input_length_at<G, G.num_transforms - 1 - Ts>(slot)), ...);
    return result;
}

} // namespace detail

/** @brief Query the length of the i-th input dimension.
 *
 *  Uses compile-time fold expression to dispatch to the correct
 *  TransformImpl specialization (runtime loops cannot be used
 *  because TransformImpl<Type> requires a constant expression).
 *
 *  @tparam G  The transform graph (NTTP)
 *  @param i   Input dimension index (0-based)
 *  @return Length of the i-th input dimension
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr index_t input_dim_length(index_t i)
{
    index_t slot = G.input_slots[i];
    return detail::input_dim_length_dispatch<G>(slot, make_index_sequence<G.num_transforms>{});
}

} // namespace ck_tile
