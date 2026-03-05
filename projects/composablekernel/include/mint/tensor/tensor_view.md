# Mint Tensor View

## Overview

`mint::tensor::tensor_view` provides a lightweight wrapper that ties a tensor
descriptor (which defines logical-to-physical index mappings) to a memory view.
Descriptors themselves are constructed from polymorphers, so tensor views inherit
all of the morpher machinery described in the poly documentation. Because Mint is
fundamentally a morpher-first system, tensor views act as the runtime veneer over
that compile-time graph. This note explains how tensor views use morphers, what
helper utilities exist for single-morpher pipelines, and how to construct new
views safely.

Key headers:

- [`mint/include/mint/tensor/tensor_view.h`](./tensor_view.h)
- [`mint/include/mint/poly/polymorpher_helper.h`](../poly/polymorpher_helper.h)
- [`mint/include/mint/poly/z2_linear_morpher_helper.h`](../poly/z2_linear_morpher_helper.h)

## Tensor View Anatomy

A tensor view stores:

1. **Tensor descriptor (`tensor_desc_`)** – exposes `top_lengths()`,
   `calculate_bottom_index()`, and `polymorpher()`; built from a polymorpher that
   chains together morphers describing the layout.
2. **Memory view (`mem_view_`)** – pointer-like object with `operator[]` used to
   access data in the chosen address space.

The descriptor must have exactly one bottom dimension (`bottom_ndim() == 1`); the
calculated bottom index identifies the physical offset used when indexing the
memory view.

## Index Propagation

When you call `tensor_view::element(idx)` or `calculate_offset(idx)`:

1. The logical index `idx` is passed to the descriptor.
2. The descriptor’s polymorpher propagates the index through its morpher stack
   (fundamental, fused, Z2, etc.).
3. The final bottom index (size 1) is returned and used to index the memory view.

Because descriptors are driven by morphers, you can treat tensor views as a
runtime façade over compile-time layout transformations.

## Transform Utilities

Tensor descriptors may be built from any combination of fundamental, fused, or
linear morphers. Generic propagation (e.g. `element`, `calculate_offset`) works
for all of them because it delegates to the polymorpher attached to the
descriptor.

A subset of helpers focuses on descriptors whose polymorpher reduces to a single
linear morpher. In the current implementation these convenience wrappers are
implemented for the `z2_linear` family but the data flow applies to any morpher
that satisfies the same compile-time constraints. The helpers
`reshape_logical`, `reorder_logical`, and `swizzle_logical` enforce this
requirement at compile time before applying the transformation.

Internally these helpers reuse `transform_tensor_view_z2`:

```cpp
auto new_polymorpher = transform_z2_polymorpher(
    tensor_view.tensor_desc_.polymorpher(),
    TransformOp{},
    args...);
auto desc = make_tensor_descriptor(new_polymorpher);
return make_tensor_view(desc, tensor_view.memory_view());
```

Thus reshaping or reordering a tensor view produces a new descriptor/memory-view
pair with updated layout while preserving the data pointer. For pipelines that
mix linear morphers with fundamental ones (e.g., `split` + Z-order swizzle), the
helpers still apply as long as the polymorpher collapses to a single morpher for
the logical dimensions under transformation.

## Building Tensor Views

1. **Create morphers** describing the desired layout (e.g. pass-through + Z2
   swizzle) and assemble them into a polymorpher.
2. **Call `make_tensor_descriptor(polymorpher)`** to obtain the descriptor.
3. **Wrap memory** with an appropriate view (e.g. pointer wrapper, pitched buffer
   view).
4. **Call `make_tensor_view(desc, mem_view)`** to produce either a mutable or
   const tensor view based on the memory view’s constness.

## API Reference

| Symbol | Summary | Notes |
|--------|---------|-------|
| `tensor_view<TensorDesc, MemoryView>` | Mutable tensor abstraction. | Inherits from internal `tensor_view_impl`. Requires `TensorDesc::bottom_ndim() == 1` and non-const memory view. |
| `const_tensor_view<TensorDesc, MemoryView>` | Read-only variant. | Constructed when the memory view advertises `is_const_memory_view() == true`. |
| `make_tensor_view(desc, mem_view)` | Factory that chooses mutable vs const view. | Overloads selected via memory view concept checks; returns the appropriate specialization. |
| `calculate_offset(idx)` | Converts a logical index into the underlying linear offset. | Accepts `nd_index<ndim()>`. Uses the descriptor’s polymorpher for propagation. |
| `element(idx)` | Provides reference access by logical index. | Mutable version returns `value_type&`; const version returns `const value_type&`. |
| `element(coordinate)` | Access via `tensor::coordinate`, which already caches bottom index. | Avoids recomputing the propagation when iterating with coordinates. |
| `tensor_desc()` | Accessor for the descriptor object. | Useful for chaining additional transformations or inspecting metadata. |
| `memory_view()` | Returns the wrapped memory view. | Allows interop with APIs expecting the original storage handle. |
| `address_space()` | Static query for the memory address space. | Mirrors the memory view’s address space (`global`, `shared`, etc.). |
| `is_tensor_view<T>` | Concept that detects tensor view types. | Used to constrain helper templates. |
| `is_single_z2_morpher_tensor_view<T>` | Concept ensuring the descriptor collapses to one linear morpher (implemented via the `z2_linear` family today). | Required by reshape/reorder/swizzle helpers. |

### Construction Pattern

```cpp
auto desc = make_tensor_descriptor(polymorpher);
auto mem = make_strided_memory_view(ptr, stride);
auto view = make_tensor_view(desc, mem); // mutable or const depending on mem
```

### Access Pattern

```cpp
nd_index<2> logical{{row, col}};
auto& ref = view.element(logical);
ref = 0; // writes back through the morpher pipeline

coordinate coord{view.tensor_desc(), logical};
float value = view.element(coord); // reuses cached bottom index
```

## Integration Tips

- **Single-morpher requirement:** Use the transform helpers only when the
  descriptor’s polymorpher collapses to a single linear morpher; otherwise the
  compile-time check will reject the call.
- **Polymorpher reuse:** Tensor descriptors expose their polymorpher. You can
  feed it into helper utilities (like `transform_z2_polymorpher`) or chain the
  descriptor within a larger polymorpher if you need to extend the layout.
- **Debugging:** `tensor_view.memory_view()` and `tensor_view.tensor_desc()` are
  accessible—use the descriptor’s `print()` method to inspect the morpher stack.

## References

- [`mint/include/mint/poly/polymorpher_doc.md`](../poly/polymorpher_doc.md) –
  Details on fused and polymorphers.
- [`mint/include/mint/poly/fundamental_morpher.md`](../poly/fundamental_morpher.md)
  – Primitive morphers that feed into tensor descriptors.
- [`mint/include/mint/poly/z2_morpher.md`](../poly/z2_morpher.md) – Z2 morpher
  background required for the tensor view transform helpers.
- [`mint/include/mint/tensor/distributed_tensor.md`](./distributed_tensor.md)
  – Cooperative tiles staged from tensor views.
- [`mint/include/mint/tensor/tiler.md`](./tiler.md) – Interface tying loop
  indices to tile windows over tensor views.
- [`mint/test/unit_test/tensor/test_tensor_view.cu`](../../test/unit_test/tensor/test_tensor_view.cu)
  – Example usage and regression tests covering tensor view transformations.
