// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Binding types and graph-construction DSL for the transform subsystem.
///
/// This header provides the user-facing verbs (`read`, `write`, `inputs`,
/// `outputs`, `transform`) and aggregate types (`TransformBinding`,
/// `GraphBinding`, `GraphInputs`, `GraphOutputs`) consumed by
/// `make_transform_graph`. A single call site composes a coordinate transform
/// with the slot routing that determines how it is wired into the surrounding
/// graph.
///
/// Example:
/// @code
/// constexpr auto g = make_transform_graph(
///     outputs(0),
///     transform(desc,                         read(1, 2, 3), write(0)),
///     transform(make_pass_through(MPerBlock), read(4),       write(2)),
///     transform(make_merge(8, 8),             read(5),       write(1, 3)),
///     inputs(4, 5));
/// @endcode

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/transform/coordinate_transform.hpp"
#include "ck_tile/experimental/core/transform/make_transform.hpp"
#include "ck_tile/experimental/core/transform/transform_graph.hpp"

#include <type_traits>

namespace ck_tile::core::transform {

/** @brief Create a DimIds routing array with -1 sentinel padding.
 *
 *  Produces a MAX_TENSOR_DIMS-element DimIds array for slot routing in
 *  transform graphs. Unused slots are set to -1 (sentinel). Called by
 *  `read()`, `write()`, `inputs()`, and `outputs()`.
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

/** @brief Specify which slots a transform reads from during traversal.
 *
 *  During traversal (user → memory), the transform pulls input values from
 *  these slots before producing its outputs.
 *
 *  @param ids  Slot indices (variadic)
 *  @return DimIds array with -1 padding for unused slots
 */
template <typename... Ts>
constexpr DimIds read(Ts... ids)
{
    return dim_ids(ids...);
}

/** @brief Specify which slots a transform writes to during traversal.
 *
 *  During traversal (user → memory), the transform deposits its output
 *  values into these slots after the read+compute step.
 *
 *  @param ids  Slot indices (variadic)
 *  @return DimIds array with -1 padding for unused slots
 */
template <typename... Ts>
constexpr DimIds write(Ts... ids)
{
    return dim_ids(ids...);
}

/// Tag wrapper declaring the input slots of a TransformGraph being built.
struct GraphInputs
{
    DimIds slots{};
    constexpr bool operator==(const GraphInputs&) const = default;
};

/// Tag wrapper declaring the output slots of a TransformGraph being built.
struct GraphOutputs
{
    DimIds slots{};
    constexpr bool operator==(const GraphOutputs&) const = default;
};

/** @brief Declare the input (user-facing) slots of a transform graph.
 *  @param ids  Slot indices, in user-coordinate order.
 *  @return GraphInputs tag passed to `make_transform_graph`.
 */
template <typename... Ts>
constexpr GraphInputs inputs(Ts... ids)
{
    return {dim_ids(ids...)};
}

/** @brief Declare the output (memory-facing) slots of a transform graph.
 *  @param ids  Slot indices, in output order (typically a single offset slot).
 *  @return GraphOutputs tag passed to `make_transform_graph`.
 */
template <typename... Ts>
constexpr GraphOutputs outputs(Ts... ids)
{
    return {dim_ids(ids...)};
}

/// Count the number of valid (non-sentinel) entries in a DimIds array.
constexpr index_t countValid(const DimIds& ids)
{
    index_t n = 0;
    for(index_t i = 0; i < MAX_TENSOR_DIMS; ++i)
        if(ids[i] >= 0)
            n++;
    return n;
}

/** @brief Bundles a coordinate transform with its read/write slot routing.
 *
 *  Structural NTTP — pure aggregate with defaulted ==. Created by the
 *  `transform()` verb and consumed by `make_transform_graph`.
 *
 *  Field semantics follow traversal data flow:
 *  - `xform`      The coordinate transform to apply
 *  - `read_dims`  Slots this transform reads from during traversal
 *  - `write_dims` Slots this transform writes to during traversal
 */
struct TransformBinding
{
    CoordinateTransform xform{};
    DimIds read_dims{};
    DimIds write_dims{};

    constexpr bool operator==(const TransformBinding&) const = default;
};

/** @brief Bundles a sub-graph with its read/write boundary slot routing.
 *
 *  Structural NTTP. The sub-graph's transforms are expanded inline into
 *  the parent graph during construction. `read_dims` maps to the
 *  sub-graph's input slots; `write_dims` maps to the sub-graph's output
 *  slots.
 */
struct GraphBinding
{
    detail::TransformGraph graph{};
    DimIds read_dims{};
    DimIds write_dims{};

    constexpr bool operator==(const GraphBinding&) const = default;
};

// Diagnostic stubs — intentionally undefined. Their names appear in linker
// errors when the binding contracts below are violated, giving the user a
// targeted message rather than a generic constexpr failure.
void transformErrorDescReadCountMismatch();
void transformErrorDescWriteCountNotOne();
void transformErrorGraphReadCountMismatch();
void transformErrorGraphWriteCountMismatch();

/** @brief Bind a coordinate transform with its slot routing.
 *
 *  Parameter order follows traversal data flow: read (input) → write (output).
 *
 *  @param xform      The coordinate transform
 *  @param read_dims  Slots this transform reads from during traversal
 *  @param write_dims Slots this transform writes to during traversal
 *  @return TransformBinding bundling all three
 *
 *  Example:
 *  @code
 *  transform(make_pass_through(128), read(0), write(1));
 *  transform(make_merge(8, 8),       read(1), write(0, 2));
 *  @endcode
 */
constexpr TransformBinding transform(CoordinateTransform xform, DimIds read_dims, DimIds write_dims)
{
    return {xform, read_dims, write_dims};
}

/** @brief Bind a TensorDescriptor as an Embed transform with slot routing.
 *
 *  Convenience overload: builds an Embed from the descriptor and bundles
 *  it with the supplied routing. `read_dims` count must equal `desc.ndim`;
 *  `write_dims` count must be 1.
 *
 *  @param desc       Tensor descriptor (lengths and strides)
 *  @param read_dims  Slots this transform reads from during traversal
 *  @param write_dims Slot this transform writes to (the offset slot)
 *  @return TransformBinding wrapping an Embed transform.
 */
constexpr TransformBinding
transform(const TensorDescriptor& desc, DimIds read_dims, DimIds write_dims)
{
    if(countValid(read_dims) != desc.ndim)
        transformErrorDescReadCountMismatch();
    if(countValid(write_dims) != 1)
        transformErrorDescWriteCountNotOne();
    return {make_embed(desc), read_dims, write_dims};
}

/** @brief Bind an existing TransformGraph as an inlined sub-graph.
 *
 *  `read_dims` count must equal the sub-graph's input count; `write_dims`
 *  count must equal the sub-graph's output count. The sub-graph's
 *  transforms are expanded into the surrounding graph during construction.
 *
 *  @param g          The sub-graph to embed
 *  @param read_dims  Parent-graph slots feeding the sub-graph's inputs
 *  @param write_dims Parent-graph slots receiving the sub-graph's outputs
 *  @return GraphBinding to be passed to `make_transform_graph`.
 */
constexpr GraphBinding
transform(const detail::TransformGraph& g, DimIds read_dims, DimIds write_dims)
{
    if(countValid(read_dims) != g.ndim_input)
        transformErrorGraphReadCountMismatch();
    if(countValid(write_dims) != g.ndim_output)
        transformErrorGraphWriteCountMismatch();
    return {g, read_dims, write_dims};
}

// NTTP-eligibility canaries — fire at compile time on any regression.
static_assert(std::is_aggregate_v<TransformBinding>);
static_assert(std::is_aggregate_v<GraphBinding>);
static_assert(std::is_aggregate_v<GraphInputs>);
static_assert(std::is_aggregate_v<GraphOutputs>);
static_assert(std::is_trivially_copyable_v<TransformBinding>);
static_assert(std::is_trivially_copyable_v<GraphBinding>);
static_assert(std::is_trivially_copyable_v<GraphInputs>);
static_assert(std::is_trivially_copyable_v<GraphOutputs>);
namespace {
constexpr TransformBinding _transform_binding_nttp_canary{};
constexpr GraphBinding _graph_binding_nttp_canary{};
constexpr GraphInputs _graph_inputs_nttp_canary{};
constexpr GraphOutputs _graph_outputs_nttp_canary{};
static_assert(_transform_binding_nttp_canary == _transform_binding_nttp_canary);
static_assert(_graph_binding_nttp_canary == _graph_binding_nttp_canary);
static_assert(_graph_inputs_nttp_canary == _graph_inputs_nttp_canary);
static_assert(_graph_outputs_nttp_canary == _graph_outputs_nttp_canary);
} // namespace

} // namespace ck_tile::core::transform
