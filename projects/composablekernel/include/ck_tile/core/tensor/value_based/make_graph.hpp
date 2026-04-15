// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file make_graph.hpp
 *  @brief Factory functions for constructing TransformGraph instances.
 *
 *  These are plain constexpr functions (no __host__ __device__) because they
 *  run only at compile time during graph construction. The resulting
 *  TransformGraph values flow to device code via NTTP.
 *
 *  Every factory calls canonicalize() before returning to guarantee
 *  NTTP deduplication safety (all unused slots zeroed).
 */

#pragma once

#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/value_based/transform_graph.hpp"
#include "ck_tile/core/tensor/value_based/make_transform.hpp"

namespace ck_tile {

/** @brief Zero-fill all unused array slots to guarantee NTTP deduplication.
 *
 *  Two graphs with identical meaningful content but different garbage in
 *  unused slots would be different NTTP values, causing duplicate template
 *  instantiations. This function ensures all unused slots are zeroed.
 *
 *  Called automatically by all graph factory functions.
 *
 *  @param g  Graph to canonicalize (modified in place)
 */
constexpr void canonicalize(TransformGraph& g)
{
    for(index_t t = g.num_transforms; t < MAX_TRANSFORMS; ++t)
    {
        g.transforms[t]     = CoordinateTransform{};
        g.t_input_slots[t]  = DimIds{};
        g.t_output_slots[t] = DimIds{};
    }
    for(index_t i = g.ndim_input; i < MAX_IO_DIMS; ++i)
    {
        g.input_slots[i] = 0;
    }
    for(index_t i = g.ndim_output; i < MAX_IO_DIMS; ++i)
    {
        g.output_slots[i] = 0;
    }
    for(index_t i = g.num_slots; i < MAX_SLOTS; ++i)
    {
        g.guaranteed_vector_lengths[i] = 0;
        g.guaranteed_vector_strides[i] = 0;
    }
}

/** @brief Create a graph for an N-D tensor with explicit strides.
 *
 *  Produces a single Embed transform: N input dims -> 1 output dim.
 *  The output is a linear memory offset computed as sum(input[i] * stride[i]).
 *
 *  Slot layout:
 *    slot 0 = output (memory offset)
 *    slots 1..N = input dimensions
 *
 *  @tparam NDim  Number of tensor dimensions (deduced from array size)
 *  @param lengths  Size of each dimension
 *  @param strides  Stride of each dimension (in elements)
 *  @return TransformGraph with 1 Embed transform, ndim_input=NDim, ndim_output=1
 *
 *  Example: make_strided_graph({8, 128, 8}, {1032, 8, 1})
 *    -> graph mapping (d0, d1, d2) to d0*1032 + d1*8 + d2*1
 */
template <index_t NDim>
constexpr TransformGraph make_strided_graph(const static_array<index_t, NDim>& lengths,
                                            const static_array<index_t, NDim>& strides)
{
    static_assert(NDim <= MAX_DIMS_PER_TRANSFORM,
                  "make_strided_graph: too many dims (max MAX_DIMS_PER_TRANSFORM)");
    static_assert(NDim >= 1, "make_strided_graph: need at least 1 dim");
    static_assert(NDim <= MAX_IO_DIMS,
                  "make_strided_graph: too many dims for graph input (max MAX_IO_DIMS)");

    TransformGraph g{};

    // Create Embed transform
    g.transforms[0]  = make_embed(lengths, strides);
    g.num_transforms = 1;

    // Routing: Embed reads from slots 1..NDim, writes to slot 0
    g.t_output_slots[0][0] = 0; // output slot = 0 (memory offset)
    for(index_t i = 0; i < NDim; ++i)
    {
        g.t_input_slots[0][i] = 1 + i;
    }

    // Graph endpoints
    g.ndim_input  = NDim;
    g.ndim_output = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        g.input_slots[i] = 1 + i; // input dims map to slots 1..NDim
    }
    g.output_slots[0] = 0; // output dim maps to slot 0

    g.num_slots = 1 + NDim; // slot 0 + NDim input slots

    // Compute element_space_size = 1 + sum((length[i] - 1) * stride[i])
    index_t ess = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        ess += (lengths[i] - 1) * strides[i];
    }
    g.element_space_size = ess;

    // Initialize guaranteed vector info to -1 (unknown)
    for(index_t i = 0; i < MAX_SLOTS; ++i)
    {
        g.guaranteed_vector_lengths[i] = -1;
        g.guaranteed_vector_strides[i] = -1;
    }

    canonicalize(g);
    return g;
}

/** @brief Create a graph for a packed (row-major) N-D tensor.
 *
 *  Produces a single Embed transform: N input dims -> 1 output dim.
 *  Computes row-major strides automatically from lengths.
 *
 *  Slot layout:
 *    slot 0 = output (memory offset)
 *    slots 1..N = input dimensions
 *
 *  @tparam NDim  Number of tensor dimensions (deduced from array size)
 *  @param lengths  Size of each dimension (row-major: last dim is contiguous)
 *  @return TransformGraph with ndim_input=NDim, ndim_output=1
 */
template <index_t NDim>
constexpr TransformGraph make_packed_graph(const static_array<index_t, NDim>& lengths)
{
    static_assert(NDim <= MAX_DIMS_PER_TRANSFORM,
                  "make_packed_graph: too many dims (max MAX_DIMS_PER_TRANSFORM)");
    static_assert(NDim >= 1, "make_packed_graph: need at least 1 dim");

    // Compute row-major strides: stride[i] = product(lengths[i+1..N-1])
    static_array<index_t, NDim> strides{};
    strides[NDim - 1] = 1;
    for(index_t i = NDim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * lengths[i + 1];
    }

    return make_strided_graph(lengths, strides);
}

/** @brief Extend a graph with new transforms.
 *
 *  Appends new transforms to an existing graph, computing internal slot
 *  routing automatically. The new transforms consume existing input dimensions
 *  and produce new input dimensions.
 *
 *  @tparam NumNewTransforms  Number of transforms to add (deduced)
 *  @param graph        The existing graph to extend
 *  @param transforms   New transforms to append
 *  @param input_dims   For each new transform: which existing input dims it reads.
 *                      Use dims() helper for padding. E.g., dims(1) or dims(0, 2).
 *                      Unused entries = -1 (set by dims()).
 *  @param output_dims  For each new transform: which new input dim indices it produces.
 *                      E.g., dims(0) means this transform produces new input dim 0.
 *                      Unused entries = -1.
 *  @return New graph with transforms appended, routing computed, and canonicalized.
 *
 *  @note element_space_size is inherited unchanged from the input graph.
 *        Transforms remap coordinates within the existing element space --
 *        they do not change the underlying buffer size.
 *
 *  Example:
 *    constexpr auto g0 = make_strided_graph({8, 128, 8}, {1032, 8, 1});
 *    constexpr auto g = apply_transforms(g0,
 *        static_array<CoordinateTransform, 2>{
 *            make_pass_through(128),
 *            make_merge(static_array<index_t, 2>{8, 8})},
 *        static_array<DimIds, 2>{dims(1), dims(0, 2)},
 *        static_array<DimIds, 2>{dims(0), dims(1)});
 */
template <index_t NumNewTransforms>
constexpr TransformGraph
apply_transforms(const TransformGraph& graph,
                 const static_array<CoordinateTransform, NumNewTransforms>& new_transforms,
                 const static_array<DimIds, NumNewTransforms>& input_dims,
                 const static_array<DimIds, NumNewTransforms>& output_dims)
{
    static_assert(NumNewTransforms >= 1, "apply_transforms: need at least 1 transform");

    TransformGraph g = graph;

    // Validate capacity
    // Note: can't use static_assert on graph.num_transforms (runtime value in constexpr)
    // The constexpr evaluation will fail naturally if bounds are exceeded.

    index_t next_slot = g.num_slots;

    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        index_t t_idx = g.num_transforms + i;

        g.transforms[t_idx] = new_transforms[i];

        // Map output slots: route to the OLD graph's input slots.
        // input_dims[i] indexes into the OLD graph's input dims.
        // The new transform's outputs feed the old graph's inputs.
        for(index_t d = 0; d < new_transforms[i].ndim_output; ++d)
        {
            index_t old_input_dim      = input_dims[i][d];
            g.t_output_slots[t_idx][d] = graph.input_slots[old_input_dim];
        }

        // Map input slots: assign fresh slots for each input dim.
        // These fresh slots receive the user's new coordinate values.
        for(index_t d = 0; d < new_transforms[i].ndim_input; ++d)
        {
            g.t_input_slots[t_idx][d] = next_slot++;
        }
    }

    g.num_transforms = graph.num_transforms + NumNewTransforms;
    g.num_slots      = next_slot;

    // Rebuild input_slots from output_dims mapping.
    // output_dims[i][d] maps the d-th input of new transform i to a new user-facing dim.
    // Since new transforms read from fresh input slots, the user's coordinate goes there.
    index_t new_ndim_input = 0;
    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        for(index_t d = 0; d < new_transforms[i].ndim_input; ++d)
        {
            if(output_dims[i][d] >= 0)
            {
                index_t new_dim_idx = output_dims[i][d];
                if(new_dim_idx + 1 > new_ndim_input)
                {
                    new_ndim_input = new_dim_idx + 1;
                }
            }
        }
    }

    // Assign new input slots: map user-facing dims to the fresh slots
    // that new transforms read from.
    static_array<index_t, MAX_IO_DIMS> new_input_slots{};
    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        for(index_t d = 0; d < new_transforms[i].ndim_input; ++d)
        {
            if(output_dims[i][d] >= 0)
            {
                index_t new_dim_idx          = output_dims[i][d];
                index_t t_idx                = graph.num_transforms + i;
                new_input_slots[new_dim_idx] = g.t_input_slots[t_idx][d];
            }
        }
    }

    g.ndim_input  = new_ndim_input;
    g.input_slots = new_input_slots;

    // output_slots and ndim_output remain unchanged from original graph
    // (transforms remap coordinates, they don't change the final output)

    // Inherit element_space_size unchanged
    // (transforms remap within existing element space)

    // Extend guaranteed vector info for new slots (initialize to -1 = unknown)
    for(index_t i = graph.num_slots; i < next_slot; ++i)
    {
        g.guaranteed_vector_lengths[i] = -1;
        g.guaranteed_vector_strides[i] = -1;
    }

    canonicalize(g);
    return g;
}

} // namespace ck_tile
