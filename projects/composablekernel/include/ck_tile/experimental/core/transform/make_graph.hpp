// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Public factories for constructing TransformGraph instances.
///
/// Three overloads of `make_transform_graph` cover every supported pattern:
///
/// 1. `make_transform_graph(desc)` — single TensorDescriptor → Embed graph.
///    N inputs (descriptor dims), 1 output (memory offset). Inputs and
///    outputs are implicit because the Embed's interface is fully determined
///    by the descriptor.
///
/// 2. `make_transform_graph(graph)` — identity, returns a copy. Useful in
///    templated code that may receive either a descriptor or a graph.
///
/// 3. `make_transform_graph(outputs(...), bindings..., inputs(...))` —
///    full explicit construction. `outputs(...)` and `inputs(...)` declare
///    the graph's interface; bindings use global slot indices via
///    `read()` / `write()`. Supports `transform(xform, ...)`,
///    `transform(desc, ...)`, and `transform(graph, ...)` for sub-graph
///    embedding. Required for any multi-transform graph.

#pragma once

#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/transform/graph_construction.hpp"
#include "ck_tile/experimental/core/transform/make_transform.hpp"
#include "ck_tile/experimental/core/transform/transform_binding.hpp"
#include "ck_tile/experimental/core/transform/transform_graph.hpp"

namespace ck_tile::core::transform {

/** @brief Build an Embed-only graph from a TensorDescriptor.
 *
 *  The descriptor's N dims become the graph's N inputs; the memory offset
 *  is the single output. No additional transforms are applied. Inputs and
 *  outputs are derived from the descriptor — no explicit declaration needed.
 *
 *  @param desc  Tensor descriptor (lengths and strides).
 *  @return TransformGraph with one Embed transform.
 */
constexpr detail::TransformGraph make_transform_graph(const TensorDescriptor& desc)
{
    return detail::make_transform_graph(desc);
}

/** @brief Identity overload — returns a copy of the input graph.
 *
 *  Provided so generic templated code can accept either a TensorDescriptor
 *  or a TransformGraph and route through `make_transform_graph(...)`
 *  uniformly.
 */
constexpr detail::TransformGraph make_transform_graph(const detail::TransformGraph& g) { return g; }

/** @brief Build a graph from explicit outputs, bindings, and inputs.
 *
 *  Required for any graph with more than one transform. The first argument
 *  is `outputs(...)`, followed by one or more transform/sub-graph bindings,
 *  terminated by `inputs(...)`.
 *
 *  Bindings can be any of:
 *  - `transform(xform, read(...), write(...))`  — single coordinate transform
 *  - `transform(desc,  read(...), write(...))`  — descriptor as Embed
 *  - `transform(graph, read(...), write(...))`  — sub-graph inlined
 *
 *  @code
 *  constexpr auto g = make_transform_graph(
 *      outputs(0),
 *      transform(desc,                         read(1, 2, 3), write(0)),
 *      transform(make_pass_through(MPerBlock), read(4),       write(2)),
 *      transform(make_merge(8, 8),             read(5),       write(1, 3)),
 *      inputs(4, 5));
 *  @endcode
 *
 *  @param outs  Graph outputs (slot indices in output order).
 *  @param args  Bindings followed by `inputs(...)`.
 *  @return Fully wired TransformGraph.
 */
template <typename... Args>
constexpr detail::TransformGraph make_transform_graph(GraphOutputs outs, Args... args)
{
    static_array<TransformBinding, MAX_TRANSFORMS> arr{};
    return detail::collectBindingsAndBuild(outs, arr, 0, 0, args...);
}

} // namespace ck_tile::core::transform
