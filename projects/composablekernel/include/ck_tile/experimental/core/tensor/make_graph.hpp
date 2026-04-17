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
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/tensor/transform_graph.hpp"
#include "ck_tile/experimental/core/tensor/make_transform.hpp"

namespace ck_tile {

static_assert(MAX_TENSOR_DIMS <= MAX_IO_DIMS, "MAX_TENSOR_DIMS must not exceed MAX_IO_DIMS");

namespace detail {

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
}

/** @brief Create a transform graph from a tensor descriptor.
 *
 *  Produces a single Embed transform that maps the descriptor's dimensions
 *  to a 1D memory offset using its lengths and strides.
 *
 *  This is the bridge between tensor metadata (TensorDescriptor) and
 *  coordinate mapping (TransformGraph). The descriptor describes WHAT
 *  the tensor looks like; the graph describes HOW to compute offsets.
 *
 *  @param desc  Tensor descriptor with lengths and strides
 *  @return TransformGraph with 1 Embed transform, ndim_upper=desc.ndim, ndim_lower=1
 *
 *  Example:
 *    constexpr auto desc = make_tensor_descriptor({8, 128, 8}, {1032, 8, 1});
 *    constexpr auto graph = make_transform_graph(desc);
 */
constexpr detail::TransformGraph make_transform_graph(const TensorDescriptor& desc)
{
    detail::TransformGraph g{};

    // Create Embed transform from the descriptor's lengths and strides.
    // NOTE: This intentionally mirrors make_embed() logic rather than calling it,
    // because make_embed is templated on NDim (compile-time) while desc.ndim is
    // a constexpr value. Cross-reference make_embed() in make_transform.hpp if
    // modifying this code.
    CoordinateTransform embed{};
    embed.type       = TransformType::EMBED;
    embed.ndim_upper = desc.ndim;
    embed.ndim_lower = 1;

    for(index_t i = 0; i < desc.ndim; ++i)
    {
        embed.lengths[i]      = desc.lengths[i];
        embed.coefficients[i] = desc.strides[i];
    }

    // Check bijectivity: stride[k] >= stride[k+1] * length[k+1]
    embed.is_bijective = true;
    for(index_t i = 0; i < desc.ndim - 1; ++i)
    {
        if(desc.strides[i] < desc.strides[i + 1] * desc.lengths[i + 1])
        {
            embed.is_bijective = false;
        }
        embed.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(desc.strides[i]));
    }

    g.transforms[0]  = embed;
    g.num_transforms = 1;

    // Routing: Embed reads from slots 1..N, writes to slot 0
    g.t_lower_slots[0][0] = 0;
    for(index_t i = 0; i < desc.ndim; ++i)
    {
        g.t_upper_slots[0][i] = 1 + i;
    }

    // Graph endpoints
    g.ndim_upper = desc.ndim;
    g.ndim_lower = 1;
    for(index_t i = 0; i < desc.ndim; ++i)
    {
        g.upper_slots[i] = 1 + i;
    }
    g.lower_slots[0] = 0;

    g.num_slots = 1 + desc.ndim;

    detail::canonicalize(g);
    return g;
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
 */
template <index_t NumNewTransforms>
constexpr detail::TransformGraph
applyTransforms(const detail::TransformGraph& graph,
                const static_array<CoordinateTransform, NumNewTransforms>& new_transforms,
                const static_array<DimIds, NumNewTransforms>& input_dims,
                const static_array<DimIds, NumNewTransforms>& output_dims)
{
    static_assert(NumNewTransforms >= 1, "applyTransforms: need at least 1 transform");

    detail::TransformGraph g = graph;

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
            index_t old_upper_dim     = input_dims[i][d];
            g.t_lower_slots[t_idx][d] = graph.upper_slots[old_upper_dim];
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

    detail::canonicalize(g);
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
template <detail::TransformGraph G>
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

} // namespace detail
} // namespace ck_tile
