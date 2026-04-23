// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Definitions for graph-construction algorithms declared in
///        `graph_construction.hpp`.
///
/// Auto-included from the bottom of `graph_construction.hpp`. Do not
/// include directly.
// IWYU pragma: private, include "ck_tile/experimental/core/transform/graph_construction.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/graph_construction.hpp"

namespace ck_tile::core::transform::detail {

constexpr void canonicalize(TransformGraph& g)
{
    for(index_t t = g.num_transforms; t < MAX_TRANSFORMS; ++t)
    {
        g.transforms[t]     = CoordinateTransform{};
        g.t_input_slots[t]  = DimIds{};
        g.t_output_slots[t] = DimIds{};
    }
    for(index_t i = g.ndim_input; i < MAX_TENSOR_DIMS; ++i)
        g.input_slots[i] = 0;
    for(index_t i = g.ndim_output; i < MAX_TENSOR_DIMS; ++i)
        g.output_slots[i] = 0;
}

constexpr TransformGraph make_transform_graph(const TensorDescriptor& desc)
{
    TransformGraph g{};

    g.transforms[0]  = make_embed(desc);
    g.num_transforms = 1;

    g.t_output_slots[0][0] = 0;
    for(index_t i = 0; i < desc.ndim; ++i)
        g.t_input_slots[0][i] = 1 + i;

    g.ndim_input  = desc.ndim;
    g.ndim_output = 1;
    for(index_t i = 0; i < desc.ndim; ++i)
        g.input_slots[i] = 1 + i;
    g.output_slots[0] = 0;

    g.num_slots = 1 + desc.ndim;

    canonicalize(g);
    return g;
}

constexpr TransformGraph
buildGraphWithIO(const static_array<TransformBinding, MAX_TRANSFORMS>& bindings,
                 index_t num_bindings,
                 const GraphInputs& ins,
                 const GraphOutputs& outs)
{
    TransformGraph g{};

    index_t max_slot = 0;
    for(index_t i = 0; i < num_bindings; ++i)
    {
        g.transforms[i] = bindings[i].xform;

        for(index_t d = 0; d < bindings[i].xform.ndim_output; ++d)
        {
            g.t_output_slots[i][d] = bindings[i].write_dims[d];
            if(bindings[i].write_dims[d] + 1 > max_slot)
                max_slot = bindings[i].write_dims[d] + 1;
        }

        for(index_t d = 0; d < bindings[i].xform.ndim_input; ++d)
        {
            g.t_input_slots[i][d] = bindings[i].read_dims[d];
            if(bindings[i].read_dims[d] + 1 > max_slot)
                max_slot = bindings[i].read_dims[d] + 1;
        }
    }

    g.num_transforms = num_bindings;
    g.num_slots      = max_slot;

    static_array<index_t, MAX_SLOTS> write_count{};
    static_array<index_t, MAX_SLOTS> is_read{};
    static_array<index_t, MAX_SLOTS> written_by{};

    for(index_t s = 0; s < MAX_SLOTS; ++s)
        written_by[s] = -1;

    for(index_t i = 0; i < num_bindings; ++i)
    {
        for(index_t d = 0; d < bindings[i].xform.ndim_output; ++d)
        {
            index_t s = bindings[i].write_dims[d];
            if(s >= 0)
            {
                write_count[s]++;
                written_by[s] = i;
            }
        }
        for(index_t d = 0; d < bindings[i].xform.ndim_input; ++d)
        {
            index_t s = bindings[i].read_dims[d];
            if(s >= 0)
                is_read[s] = 1;
        }
    }

    for(index_t s = 0; s < max_slot; ++s)
    {
        if(write_count[s] > 1)
            graphValidationErrorDoubleWriteToSlot();
    }

    for(index_t i = 0; i < num_bindings; ++i)
    {
        for(index_t d = 0; d < bindings[i].xform.ndim_input; ++d)
        {
            index_t s = bindings[i].read_dims[d];
            if(s >= 0 && write_count[s] > 0)
            {
                if(written_by[s] <= i)
                    graphValidationErrorWrongTraversalOrder();
            }
        }
    }

    index_t n_ins = countValid(ins.slots);
    for(index_t i = 0; i < n_ins; ++i)
    {
        if(write_count[ins.slots[i]] > 0)
            graphValidationErrorInputSlotIsWritten();
    }

    index_t n_outs = countValid(outs.slots);
    for(index_t i = 0; i < n_outs; ++i)
    {
        if(is_read[outs.slots[i]] && !write_count[outs.slots[i]])
            graphValidationErrorOutputSlotIsReadOnly();
    }

    g.ndim_input = n_ins;
    for(index_t i = 0; i < n_ins; ++i)
        g.input_slots[i] = ins.slots[i];

    g.ndim_output = n_outs;
    for(index_t i = 0; i < n_outs; ++i)
        g.output_slots[i] = outs.slots[i];

    canonicalize(g);
    return g;
}

constexpr index_t expandGraphBinding(static_array<TransformBinding, MAX_TRANSFORMS>& arr,
                                     index_t idx,
                                     const GraphBinding& gb,
                                     index_t slot_offset)
{
    const auto& sub = gb.graph;

    static_array<index_t, MAX_SLOTS> remap{};
    for(index_t s = 0; s < MAX_SLOTS; ++s)
        remap[s] = slot_offset + s;

    for(index_t i = 0; i < sub.ndim_input; ++i)
        remap[sub.input_slots[i]] = gb.read_dims[i];

    for(index_t i = 0; i < sub.ndim_output; ++i)
        remap[sub.output_slots[i]] = gb.write_dims[i];

    for(index_t i = 0; i < sub.num_transforms; ++i)
    {
        if(idx + i >= MAX_TRANSFORMS)
            graphValidationErrorTooManyTransforms();

        arr[idx + i].xform = sub.transforms[i];

        DimIds remapped_read{};
        for(auto& x : remapped_read.elems)
            x = -1;
        for(index_t d = 0; d < sub.transforms[i].ndim_input; ++d)
            remapped_read[d] = remap[sub.t_input_slots[i][d]];
        arr[idx + i].read_dims = remapped_read;

        DimIds remapped_write{};
        for(auto& x : remapped_write.elems)
            x = -1;
        for(index_t d = 0; d < sub.transforms[i].ndim_output; ++d)
            remapped_write[d] = remap[sub.t_output_slots[i][d]];
        arr[idx + i].write_dims = remapped_write;
    }

    return sub.num_transforms;
}

constexpr TransformGraph collectBindingsAndBuild(GraphOutputs outs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS> arr,
                                                 index_t idx,
                                                 index_t /*max_slot*/,
                                                 GraphInputs ins)
{
    return buildGraphWithIO(arr, idx, ins, outs);
}

template <typename... Rest>
constexpr TransformGraph collectBindingsAndBuild(GraphOutputs outs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS> arr,
                                                 index_t idx,
                                                 index_t max_slot,
                                                 TransformBinding binding,
                                                 Rest... rest)
{
    arr[idx] = binding;
    for(index_t d = 0; d < binding.xform.ndim_input; ++d)
        if(binding.read_dims[d] >= 0 && binding.read_dims[d] + 1 > max_slot)
            max_slot = binding.read_dims[d] + 1;
    for(index_t d = 0; d < binding.xform.ndim_output; ++d)
        if(binding.write_dims[d] >= 0 && binding.write_dims[d] + 1 > max_slot)
            max_slot = binding.write_dims[d] + 1;
    return collectBindingsAndBuild(outs, arr, idx + 1, max_slot, rest...);
}

template <typename... Rest>
constexpr TransformGraph collectBindingsAndBuild(GraphOutputs outs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS> arr,
                                                 index_t idx,
                                                 index_t max_slot,
                                                 GraphBinding gbinding,
                                                 Rest... rest)
{
    index_t n_r = countValid(gbinding.read_dims);
    index_t n_w = countValid(gbinding.write_dims);
    for(index_t i = 0; i < n_r; ++i)
        if(gbinding.read_dims[i] + 1 > max_slot)
            max_slot = gbinding.read_dims[i] + 1;
    for(index_t i = 0; i < n_w; ++i)
        if(gbinding.write_dims[i] + 1 > max_slot)
            max_slot = gbinding.write_dims[i] + 1;

    index_t added = expandGraphBinding(arr, idx, gbinding, max_slot);
    return collectBindingsAndBuild(
        outs, arr, idx + added, max_slot + gbinding.graph.num_slots, rest...);
}

} // namespace ck_tile::core::transform::detail
