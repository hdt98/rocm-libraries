// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Internal graph-construction algorithms used by `make_transform_graph`.
///
/// Houses the routing/validation algorithms that turn a sequence of
/// `TransformBinding` / `GraphBinding` values into a fully wired
/// `TransformGraph`. Not part of the user-facing API.
// IWYU pragma: private, include "ck_tile/experimental/core/transform/make_graph.hpp"
#pragma once

#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/transform/coordinate_transform.hpp"
#include "ck_tile/experimental/core/transform/make_transform.hpp"
#include "ck_tile/experimental/core/transform/transform_binding.hpp"
#include "ck_tile/experimental/core/transform/transform_graph.hpp"

namespace ck_tile::core::transform::detail {

// Diagnostic stubs — intentionally undefined. Their names appear in linker
// errors when graph-validation contracts are violated, giving the user a
// targeted message rather than a generic constexpr failure.
void graphValidationErrorDoubleWriteToSlot();
void graphValidationErrorWrongTraversalOrder();
void graphValidationErrorInputSlotIsWritten();
void graphValidationErrorOutputSlotIsReadOnly();
void graphValidationErrorTooManyTransforms();

// Zero-initialize unused slots so two graphs with identical meaningful
// content produce identical NTTP values. Without this, garbage in unused
// array tail slots would defeat C++20 structural equality and the compiler
// would generate duplicate template instantiations for logically identical
// graphs.
constexpr void canonicalize(TransformGraph& g);

// Single-descriptor graph: one Embed transform with N inputs and 1 output.
// Used by the public make_transform_graph(desc) convenience overload.
constexpr TransformGraph make_transform_graph(const TensorDescriptor& desc);

// Build a graph from bindings with explicit inputs/outputs. All read()/write()
// values are global slot positions. Validates that:
//   - no slot is written by more than one transform,
//   - traversal order is correct (a transform's reads come from earlier writes),
//   - input slots are not written, and
//   - output slots that are read elsewhere are also written.
constexpr TransformGraph
buildGraphWithIO(const static_array<TransformBinding, MAX_TRANSFORMS>& bindings,
                 index_t num_bindings,
                 const GraphInputs& ins,
                 const GraphOutputs& outs);

// Inline a sub-graph into the parent's binding array, remapping its internal
// slots into the parent's slot space.
constexpr index_t expandGraphBinding(static_array<TransformBinding, MAX_TRANSFORMS>& arr,
                                     index_t idx,
                                     const GraphBinding& gb,
                                     index_t slot_offset);

// Recursive variadic helpers that walk the
//   outputs(...), bindings..., inputs(...)
// parameter pack and dispatch into buildGraphWithIO once all bindings are
// collected.
template <typename... Rest>
constexpr TransformGraph collectBindingsAndBuild(GraphOutputs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS>,
                                                 index_t,
                                                 index_t,
                                                 TransformBinding,
                                                 Rest...);

template <typename... Rest>
constexpr TransformGraph collectBindingsAndBuild(GraphOutputs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS>,
                                                 index_t,
                                                 index_t,
                                                 GraphBinding,
                                                 Rest...);

constexpr TransformGraph collectBindingsAndBuild(GraphOutputs outs,
                                                 static_array<TransformBinding, MAX_TRANSFORMS> arr,
                                                 index_t idx,
                                                 index_t max_slot,
                                                 GraphInputs ins);

} // namespace ck_tile::core::transform::detail

#include "ck_tile/experimental/core/transform/graph_construction_impl.hpp"
