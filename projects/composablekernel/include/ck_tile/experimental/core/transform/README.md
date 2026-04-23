# `ck_tile::core::transform`

A value-based coordinate transform graph for ck_tile. Encodes the entire
transform pipeline as a single C++20 structural NTTP, eliminating the
intermediate template-type explosion characteristic of the v1 type-based
descriptor system.

## Status

Experimental. Lives under `include/ck_tile/experimental/core/transform/` for
now; namespace already matches the post-promotion path
(`include/ck_tile/core/transform/`), so promotion will be a pure file move
with no source edits.

See `build_time_value_based_transform_graph_report.md` for the design
rationale, benchmarks, and gap analysis.

## Public API

The transform module is composed as a small set of vocabulary types,
factories, and a binding DSL. Users include the headers they need (no
aggregate facade — IWYU per Style Guide §9.1).

| Header | Provides |
|---|---|
| `ck_tile/experimental/core/transform/coordinate_transform.hpp` | `CoordinateTransform`, `DimIds` |
| `ck_tile/experimental/core/transform/transform_type.hpp` | `enum struct TransformType` |
| `ck_tile/experimental/core/transform/transform_binding.hpp` | `TransformBinding`, `GraphBinding`, `GraphInputs`, `GraphOutputs`; verbs: `read`, `write`, `dim_ids`, `inputs`, `outputs`, `transform` |
| `ck_tile/experimental/core/transform/transform_graph.hpp` | `TransformGraph` (in `detail::`) and the public traversal API: `map`, `calculateOffset`, `inputDimLength`, `reverseMap`, `reverseCalculateOffset`, `isGraphBijective` |
| `ck_tile/experimental/core/transform/make_transform.hpp` | `make_pass_through`, `make_merge`, `make_unmerge`, `make_embed`, `make_pad`, `make_right_pad`, `make_xor`, `dims` |
| `ck_tile/experimental/core/transform/make_graph.hpp` | `make_transform_graph` (3 overloads) |

`TensorDescriptor` lives one level up in `ck_tile/experimental/core/tensor/tensor_descriptor.hpp`
(namespace `ck_tile::core::tensor`). The transform module re-exports
`TensorDescriptor` and `MAX_TENSOR_DIMS` into its own namespace via
using-declarations in `coordinate_transform.hpp` so consumers needn't
fully qualify them.

## `detail::` namespace

`ck_tile::core::transform::detail` holds implementation internals:
`TransformGraph`, `TransformImpl<T>` and its 6 specializations, magic
division, graph-construction algorithms, validation diagnostic stubs.

Per Style Guide §10.2: code in `detail::` is **private**. No stability
guarantees. End users should never reference `detail::` symbols directly.

The test file is a privileged consumer — it reaches into `detail::` for
white-box correctness checks (magic division, `TransformImpl` bounds
checking). New end-user code must not.

## Worked example

```cpp
#include "ck_tile/experimental/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/experimental/core/transform/make_graph.hpp"
#include "ck_tile/experimental/core/transform/make_transform.hpp"
#include "ck_tile/experimental/core/transform/transform_binding.hpp"
#include "ck_tile/experimental/core/transform/transform_graph.hpp"

using ck_tile::core::tensor::make_tensor_descriptor;
using ck_tile::core::transform::calculateOffset;
using ck_tile::core::transform::dims;
using ck_tile::core::transform::inputs;
using ck_tile::core::transform::make_merge;
using ck_tile::core::transform::make_pass_through;
using ck_tile::core::transform::make_transform_graph;
using ck_tile::core::transform::outputs;
using ck_tile::core::transform::read;
using ck_tile::core::transform::transform;
using ck_tile::core::transform::write;

constexpr ck_tile::index_t MPerBlock = 128;
constexpr ck_tile::index_t KPerBlock = 64;

constexpr auto desc = make_tensor_descriptor(
    dims(KPerBlock / 8, MPerBlock, 8),
    dims((MPerBlock + 1) * 8, 8, 1));

// Slot assignment:
//   0       offset (Embed write)
//   1, 2, 3 physical dims K/8, M, K%8 (Embed read)
//   4       user M (PassThrough read, graph input)
//   5       user K (Merge read, graph input)
constexpr auto g = make_transform_graph(
    outputs(0),
    transform(desc,                            read(1, 2, 3), write(0)),
    transform(make_pass_through(MPerBlock),    read(4),       write(2)),
    transform(make_merge(KPerBlock / 8, 8),    read(5),       write(1, 3)),
    inputs(4, 5));

constexpr ck_tile::index_t off =
    calculateOffset<g>(ck_tile::static_array<ck_tile::index_t, 2>{m_idx, k_idx});
```

The test file at `test/ck_tile/experimental/core/transform/test_transform_graph.cpp`
exercises every feature with `static_assert`s — if it compiles, all
correctness tests pass.

## Style-guide notes

- Public symbols live in `ck_tile::core::transform`. Internal symbols
  live in `ck_tile::core::transform::detail`.
- `_impl.hpp` companion files hold template definitions > 5 lines for the
  largest two headers (`transform_graph.hpp`, `graph_construction.hpp`),
  per Style Guide §15.1.
- All seven NTTP-eligible aggregates carry `static_assert` canaries
  (`std::is_aggregate_v`, `std::is_trivially_copyable_v`, defaulted `==`
  smoke test) at the bottom of their owning header — these fire at
  compile time on any regression that breaks NTTP eligibility.
- No aggregate include / facade header (Style Guide §9.1). Users include
  only what they need.
