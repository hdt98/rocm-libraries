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
#include "ck_tile/experimental/core/tensor/transform_graph.hpp"
#include "ck_tile/experimental/core/tensor/make_transform.hpp"

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
        g.transforms[t]    = CoordinateTransform{};
        g.t_upper_slots[t] = DimIds{};
        g.t_lower_slots[t] = DimIds{};
    }
    for(index_t i = g.ndim_upper; i < MAX_IO_DIMS; ++i)
    {
        g.upper_slots[i] = 0;
    }
    for(index_t i = g.ndim_lower; i < MAX_IO_DIMS; ++i)
    {
        g.lower_slots[i] = 0;
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
 *  @return TransformGraph with 1 Embed transform, ndim_upper=NDim, ndim_lower=1
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
    g.t_lower_slots[0][0] = 0; // output slot = 0 (memory offset)
    for(index_t i = 0; i < NDim; ++i)
    {
        g.t_upper_slots[0][i] = 1 + i;
    }

    // Graph endpoints
    g.ndim_upper = NDim;
    g.ndim_lower = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        g.upper_slots[i] = 1 + i; // input dims map to slots 1..NDim
    }
    g.lower_slots[0] = 0; // output dim maps to slot 0

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
 *  @return TransformGraph with ndim_upper=NDim, ndim_lower=1
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

/** @brief Push new transforms onto the graph's transform stack.
 *
 *  New transforms are pushed above the existing stack (closer to the user).
 *  During traversal, they will be popped first (LIFO). Slot routing is
 *  computed automatically.
 *
 *  See transform_graph.hpp file header for the complete explanation of how
 *  input_dims and output_dims control slot routing, with worked examples
 *  and diagrams.
 *
 *  @tparam NumNewTransforms  Number of transforms to add (deduced)
 *  @param graph        The existing graph to extend
 *  @param transforms   New transforms to append
 *  @param input_dims   Which OLD graph dims each transform REPLACES.
 *                      Controls WRITE routing during traversal.
 *  @param output_dims  Which NEW user-facing dims each transform CREATES.
 *                      Controls READ routing during traversal.
 *  @return New graph with transforms appended, routing computed, and canonicalized.
 *
 *  @note element_space_size is inherited unchanged from the input graph.
 *        Transforms remap coordinates within the existing element space --
 *        they do not change the underlying buffer size.
 */
template <index_t NumNewTransforms>
constexpr TransformGraph
applyTransforms(const TransformGraph& graph,
                const static_array<CoordinateTransform, NumNewTransforms>& new_transforms,
                const static_array<DimIds, NumNewTransforms>& input_dims,
                const static_array<DimIds, NumNewTransforms>& output_dims)
{
    static_assert(NumNewTransforms >= 1, "applyTransforms: need at least 1 transform");

    TransformGraph g = graph;

    // Validate capacity
    // Note: can't use static_assert on graph.num_transforms (runtime value in constexpr)
    // The constexpr evaluation will fail naturally if bounds are exceeded.

    index_t next_slot = g.num_slots;

    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        index_t t_idx = g.num_transforms + i;

        g.transforms[t_idx] = new_transforms[i];

        // input_dims controls WRITE routing during traversal:
        // "which old dims does this transform replace?"
        // Wire the transform's output slots to those old dim slots.
        for(index_t d = 0; d < new_transforms[i].ndim_lower; ++d)
        {
            index_t old_input_dim     = input_dims[i][d];
            g.t_lower_slots[t_idx][d] = graph.upper_slots[old_input_dim];
        }

        // Allocate fresh slots for the transform's READ side.
        // output_dims (below) will map these to user-facing dimensions.
        for(index_t d = 0; d < new_transforms[i].ndim_upper; ++d)
        {
            g.t_upper_slots[t_idx][d] = next_slot++;
        }
    }

    g.num_transforms = graph.num_transforms + NumNewTransforms;
    g.num_slots      = next_slot;

    // output_dims controls READ routing during traversal:
    // "which new user-facing dim does each transform's input become?"
    // Map user dim indices to the fresh slots allocated above.
    index_t new_ndim_upper = 0;
    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        for(index_t d = 0; d < new_transforms[i].ndim_upper; ++d)
        {
            if(output_dims[i][d] >= 0)
            {
                index_t new_dim_idx = output_dims[i][d];
                if(new_dim_idx + 1 > new_ndim_upper)
                {
                    new_ndim_upper = new_dim_idx + 1;
                }
            }
        }
    }

    // Wire user-facing dim indices to the fresh slots that transforms read from.
    static_array<index_t, MAX_IO_DIMS> new_upper_slots{};
    for(index_t i = 0; i < NumNewTransforms; ++i)
    {
        for(index_t d = 0; d < new_transforms[i].ndim_upper; ++d)
        {
            if(output_dims[i][d] >= 0)
            {
                index_t new_dim_idx          = output_dims[i][d];
                index_t t_idx                = graph.num_transforms + i;
                new_upper_slots[new_dim_idx] = g.t_upper_slots[t_idx][d];
            }
        }
    }

    g.ndim_upper  = new_ndim_upper;
    g.upper_slots = new_upper_slots;

    // lower_slots and ndim_lower remain unchanged from original graph
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

/** @brief Check whether all transforms in a graph are bijective.
 *
 *  A graph is reversible if and only if every transform is bijective.
 *  The is_bijective flag is set automatically by factory functions.
 *
 *  @tparam G  The transform graph (NTTP)
 *  @return true if all transforms are bijective
 */
template <TransformGraph G>
constexpr bool isGraphBijective()
{
    for(index_t t = 0; t < G.num_transforms; ++t)
    {
        if(!G.transforms[t].is_bijective)
        {
            return false;
        }
    }
    return true;
}

} // namespace ck_tile
