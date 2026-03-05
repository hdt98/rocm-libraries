# Mint Fused and Polymorpher Overview

## Motivation

Fundamental and Z2 morphers describe small, self-contained index transformations.
In real applications we need to assemble many morphers into a larger pipeline and
reason about shared dimensions. Mint treats morpher composition as the core
abstraction for every higher-level facility—tensor views, distributed tensors,
and tile iterators are thin veneers over these graphs. Each polymorpher makes the
logical-to-physical mapping explicit by wiring the output (bottom) coordinates of
one stage to the input (top) coordinates of the next. The `fused_morpher` and
`polymorpher` templates provide the glue: they track how component morphers share
dimensions, enforce acyclic dependencies, and expose unified propagation routines
that operate across the entire morpher stack.

All functionality discussed here lives in:

- [`mint/include/mint/poly/fused_morpher.h`](./fused_morpher.h)
- [`mint/include/mint/poly/polymorpher.h`](./polymorpher.h)
- [`mint/include/mint/poly/polymorpher_helper.h`](./polymorpher_helper.h)

## Fused Morphers

`fused_morpher<Morphers, kDimPairs>` combines a compile-time tuple of morphers
(`Morphers`) into a single CRTP morpher. Each component morpher contributes its
top/bottom dimensions, while `kDimPairs` declares how those dimensions connect
between morphers. The template unifies several responsibilities:

1. **Dimension deduplication:**
   - Converts each `(morpher_id, local_dim)` pair into a unique global dimension.
   - Computes `top_dims_`, `bottom_dims_`, and `paired_dims_` across the fused
     graph.

2. **Dependency ordering:**
   - Builds a bipartite graph from `kDimPairs` to determine which morphers must
     run before others during top-down and bottom-up propagation.
   - Enforces acyclicity at compile time via `static_assert`.

3. **Linearity propagation:**
   - Aggregates `is_linear_top_down_` / `is_linear_bottom_up_` flags from child
     morphers to allow downstream optimizations.

### Propagation

Fused morphers execute each component in the correct order while slicing the
shared index into per-morpher subsets:

```cpp
static_for_n<num_morpher_>()([this, &idx](auto i) {
  constexpr index_t morpher_id = top_down_sorted_morpher_ids()[i];
  constexpr auto dims = morpher_unique_dims<morpher_id>();
  auto local_idx = idx.template get_subset<dims.size(), dims>();
  morphers_.template at<morpher_id>().propagate_index_top_down(local_idx);
  idx.template set_subset<dims.size(), dims>(local_idx);
});
```

Bottom-up and delta propagation follow the same pattern using reversed order.

### When to Use

- Build multi-stage tiling/transposition pipelines without manually managing
  index ownership.
- Fuse heterogeneous morphers (fundamental, Z2, custom) where dimensions are
  shared or passed from one stage to another.

## Polymorphers

`polymorpher<Morphers, kDimPairs, AllLengths, kMorpherAliases, kDimAliases>`
extends `fused_morpher` with user-friendly metadata:

- **Aliases:** Arrays that map stable identifiers to morphers and dimensions.
  Useful when different subsystems refer to the same dimension using semantic
  names.
- **Lengths:** Stores the full set of dimension lengths (`AllLengths`) aligned
  with the deduplicated dimension order.
- **Lookup tables:** Convenience methods such as `alias_to_morpher()` and
  `alias_to_dim()` provide O(1) constexpr lookups from aliases back to fused
  indices.

Polymorphers expose the entire `fused_morpher` API while adding helpers to query
alias mappings or print the assembled structure.

### Construction

Most call sites rely on `make_polymorpher` from
[`mint/include/mint/poly/polymorpher_helper.h`](./polymorpher_helper.h). The
helper validates alias coverage, constructs the alias arrays, and computes the
ordered length vector before returning a `polymorpher` instance.

```cpp
constexpr auto pm = make_polymorpher<
    kDimPairs,
    kAliasToMorpher,
    kAliasToMorpherLocalDim>(morphers_tuple, dim_alias_ordered_lengths);
```

### Typical Workflow

1. **Author component morphers.** Prepare fundamental or Z2 morphers that capture
   local transforms.
2. **Describe connections.** Populate `kDimPairs` to indicate how bottom dims of
   one morpher feed into top dims of another.
3. **Assign aliases.** Provide readable names (enums or integral constants) for
   morphers and dimensions.
4. **Instantiate polymorpher.** Use `make_polymorpher` to obtain the assembled
   object.
5. **Propagate indices.** Call fused-polymorpher propagation APIs just like any
   other morpher. Aliases enable downstream code to refer to dimensions without
   memorizing positional indices.

## Integration Tips

- **Dimension graphs:** Ensure `kDimPairs` express a DAG; cycles trigger compile
  errors. If a cycle is intentional, break it by restructuring morphers so that
  shared dimensions become paired dimensions.
- **Aliasing strategy:** Prefer strongly typed enums for aliases. They offer
  better compile-time diagnostics than raw integers.
- **Interoperability:** Polymorphers can be chained with other morphers using
  helper utilities (e.g., turn a polymorpher into a single stage within a larger
  fused morpher).
- **Debugging:** Use the `print()` member to inspect morphers, aliases, and
  lengths at runtime when debugging GPU kernels.

## Related References

- [`mint/include/mint/poly/fundamental_morpher.md`](./fundamental_morpher.md)
  – Primitive building blocks.
- [`mint/include/mint/poly/z2_morpher.md`](./z2_morpher.md)
  – Bitwise GF(2) morphers frequently fused into polymorpher stacks.
- [`mint/test/unit_test/poly/test_poly_polymorpher.cu`](../../test/unit_test/poly/test_poly_polymorpher.cu)
  – Examples of real polymorpher graphs and propagation checks.
