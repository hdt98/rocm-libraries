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
    for(index_t i = g.ndim_upper; i < MAX_TENSOR_DIMS; ++i)
    {
        g.upper_slots[i] = 0;
    }
    for(index_t i = g.ndim_lower; i < MAX_TENSOR_DIMS; ++i)
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
    using EmbedImpl = TransformImpl<TransformType::EMBED>;

    detail::TransformGraph g{};

    // Build Embed transform via Schema.
    // Cannot delegate to make_embed() because it is templated on NDim
    // (compile-time) while desc.ndim is a constexpr runtime value.
    // Both use the same Schema layout and bijectivity/magic_div logic.
    CoordinateTransform embed{};
    embed.type       = TransformType::EMBED;
    embed.ndim_upper = desc.ndim;
    embed.ndim_lower = 1;

    EmbedImpl::Schema d{};
    for(index_t i = 0; i < desc.ndim; ++i)
    {
        d.dim_lengths[i] = desc.lengths[i];
        d.strides[i]     = desc.strides[i];
    }

    embed.is_bijective = true;
    for(index_t i = 0; i < desc.ndim - 1; ++i)
    {
        if(desc.strides[i] < desc.strides[i + 1] * desc.lengths[i + 1])
            embed.is_bijective = false;
        d.magic_divs[i] = computeMagicDiv(static_cast<uint32_t>(desc.strides[i]));
    }

    EmbedImpl::writeSchema(embed, d);
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
    static_array<index_t, MAX_TENSOR_DIMS> new_upper_slots{};
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

/// Unpack N TransformBindings into parallel arrays for applyTransforms.
template <index_t N>
constexpr detail::TransformGraph applyBindings(const detail::TransformGraph& graph,
                                               const static_array<TransformBinding, N>& bindings)
{
    static_array<CoordinateTransform, N> transforms{};
    static_array<DimIds, N> lower_dims{}; // which old dims to replace
    static_array<DimIds, N> upper_dims{}; // which new dims to create

    for(index_t i = 0; i < N; ++i)
    {
        transforms[i] = bindings[i].xform;
        lower_dims[i] = bindings[i].lower_dims;
        upper_dims[i] = bindings[i].upper_dims;
    }

    // applyTransforms param order: (transforms, input_dims=lower, output_dims=upper)
    return applyTransforms(graph, transforms, lower_dims, upper_dims);
}

} // namespace detail

// ============================================================================
// User-facing make_transform_graph overloads using TransformBinding
// ============================================================================

/** @brief Create a transform graph from a descriptor and transform bindings.
 *
 *  Combines make_transform_graph(desc) + applyTransforms into a single call.
 *  All bindings are applied as a single batch (not sequentially), so lower()
 *  indices always refer to the original descriptor's dimensions.
 *
 *  @param desc      Tensor descriptor (creates the base Embed)
 *  @param bindings  Transform bindings (variadic)
 *  @return TransformGraph with base Embed + all bindings applied
 *
 *  Example:
 *    constexpr auto g = make_transform_graph(desc,
 *        transform(make_pass_through(M),  upper(0), lower(1)),
 *        transform(make_merge({K/8, 8}),  upper(1), lower(0, 2)));
 */
template <typename... Bindings>
constexpr detail::TransformGraph make_transform_graph(const TensorDescriptor& desc,
                                                      Bindings... bindings)
{
    constexpr index_t N = sizeof...(Bindings);
    auto g              = detail::make_transform_graph(desc);
    static_array<TransformBinding, N> arr{bindings...};
    return detail::applyBindings(g, arr);
}

/** @brief Create a transform graph from transform bindings only (no descriptor).
 *
 *  For general coordinate mappings that don't involve a tensor. The first
 *  binding typically provides the base transform (e.g., an Embed defining
 *  the memory layout).
 *
 *  @param first     First transform binding (typically the base/Embed)
 *  @param rest      Additional transform bindings (variadic)
 *  @return TransformGraph with all bindings applied
 *
 *  Example:
 *    constexpr auto g = make_transform_graph(
 *        transform(make_embed({8,128,8}, {1032,8,1}), upper(0,1,2), lower(0)),
 *        transform(make_pass_through(128),            upper(0),     lower(1)),
 *        transform(make_merge({8, 8}),                upper(1),     lower(0, 2)));
 */
template <typename... Bindings>
constexpr detail::TransformGraph make_transform_graph(TransformBinding first, Bindings... rest)
{
    // Build the base graph from the first binding
    detail::TransformGraph g{};

    g.transforms[0]  = first.xform;
    g.num_transforms = 1;

    // Wire based on the first binding's dims
    // Lower dims: where the transform writes (toward memory)
    index_t num_lower = 0;
    for(index_t i = 0; i < MAX_TENSOR_DIMS; ++i)
    {
        if(first.lower_dims[i] >= 0)
        {
            g.t_lower_slots[0][i] = first.lower_dims[i];
            num_lower++;
        }
    }
    g.ndim_lower = num_lower;
    for(index_t i = 0; i < num_lower; ++i)
    {
        g.lower_slots[i] = first.lower_dims[i];
    }

    // Upper dims: where the transform reads (from user side)
    index_t num_upper = 0;
    index_t max_slot  = num_lower;
    for(index_t i = 0; i < MAX_TENSOR_DIMS; ++i)
    {
        if(first.upper_dims[i] >= 0)
        {
            index_t slot                       = num_lower + i;
            g.t_upper_slots[0][i]              = slot;
            g.upper_slots[first.upper_dims[i]] = slot;
            if(slot >= max_slot)
                max_slot = slot + 1;
            num_upper++;
        }
    }
    g.ndim_upper = num_upper;
    g.num_slots  = max_slot;

    detail::canonicalize(g);

    // Apply remaining bindings as a batch
    if constexpr(sizeof...(rest) > 0)
    {
        constexpr index_t N = sizeof...(rest);
        static_array<TransformBinding, N> arr{rest...};
        g = detail::applyBindings(g, arr);
    }
    return g;
}

} // namespace ck_tile
