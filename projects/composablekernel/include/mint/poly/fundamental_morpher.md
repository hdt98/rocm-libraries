# Mint Fundamental Morphers

## Overview

Fundamental morphers provide the building blocks for index transformation in the
Mint polymorpher stack. They represent basic operations—identity, shifts,
splitting/merging, and geometric mappings—that can be composed into richer
pipelines. Because Mint’s architecture is morpher-centric, these primitives form
the foundation on which every tensor, tiler, and tile routine is expressed. Each
morpher implements a concrete mapping from logical “top” coordinates to physical
“bottom” coordinates (and often back again), making the logical-to-physical
relationship explicit at compile time.
All definitions reside in
[`mint/include/mint/poly/fundamental_morpher.h`](./fundamental_morpher.h) and use
the common CRTP base [`mint/include/mint/poly/morpher.h`](./morpher.h).

Each morpher advertises:

- Directional capabilities (`can_top_down_`, `can_bottom_up_`)
- Dimension metadata (`top_ndim_`, `bottom_ndim_`, `paired_ndim_`)
- Linearity flags for compile-time optimization
- Host/device propagation methods for indices and deltas

Fundamental morphers are typically chained with helpers from
`mint/include/mint/poly/fundamental_morpher_helper.h` or used as leaves inside
composite morphers.

## Identity and Routing Morphers

| Morpher | Description | Use Case |
|---------|-------------|----------|
| `none` | Trivial identity morpher with a single dimension; all propagation routines are no-ops. | Placeholder or default morpher when no transformation is required. |
| `pass_through` | Connects one top dimension to one bottom dimension (two-slot index tuple). | Wire indices between stages while keeping deltas coherent. |

Both morphers are fully linear in both directions and preserve indices exactly.

## Arithmetic Morphers

### `shift<ShiftValue>`

Adds (top-down) or subtracts (bottom-up) a constant offset when mapping between a
pair of dimensions. The shift value is stored as template parameter/constructor
state and participates in equality comparisons. Commonly inserted to adjust base
addresses when tiling memory.

### `project<Coefficients>`

Performs a weighted sum of multiple top dimensions into a single bottom
coordinate. Only supports top-down propagation because the reverse mapping is
not uniquely defined. Useful for computing linear indices from multi-dimensional
coordinates (e.g., Morton curve variants or custom strides).

## Dimension Reshaping Morphers

### `split<BottomLengths>`

Converts a single top index into `BottomLengths::size()` bottom dimensions. The
constructor precomputes fast divisors to accelerate quotient/remainder
calculation on both host and device. Supports propagation of indices and deltas
in both directions and handles partially frozen dimensions for speculative
updates.

### `merge<TopLengths>`

Inverse of `split`: flattens multiple top dimensions into a single bottom index.
Shares the same fast-division infrastructure and is typically chained with
`split` to express reshapes or layout changes.

## Dimensionality Adjustments

| Morpher | Direction | Purpose |
|---------|-----------|---------|
| `insert<kN>` | Bottom-up only | Append `kN` new dimensions that pass values through untouched. |
| `remove<kN>` | Top-down only | Drop `kN` dimensions from the index tuple. |
| `insert_length_one` | Bidirectional | Introduce a singleton dimension that always reads as zero on the top side. |

These morphers help orchestrate intermediate shapes in a pipeline without
altering existing coordinates.

## Geometric Morphers

### `rotate2d<RotateBound, RotateStep>`

Implements modular rotation in a 2D plane. During top-down propagation it
subtracts `idx[2] * RotateStep` from `idx[3]` (modulo `RotateBound`) to produce
the rotated coordinate pair; bottom-up performs the inverse transformation. This
morpher is typically used to express swizzles or cyclic permutations in tiled
layouts. Both directions are supported and rely on power-of-two bounds for
bitwise efficiency.

## Integration Workflow

1. **Select fundamental stages.** Choose identity, arithmetic, reshape, or
   geometric morphers based on the index manipulation required.
2. **Compose with helpers.** Use functions in
   `mint/include/mint/poly/fundamental_morpher_helper.h` to chain (`chain`),
   concatenate (`concat`), or fuse morphers with others (e.g., Z2 morphers).
3. **Attach to tensors.** Distributed tensor factories (for example,
   [`mint/include/mint/tensor/distributed_tensor.h`](../tensor/distributed_tensor.h))
   bind morphers to physical layouts for blocks, warps, or thread groups.
4. **Propagate indices.** Invoke `propagate_index_*` inside kernels or host-side
   utilities to translate between logical and physical coordinates.

## Diagnostics and Debugging

Most morphers expose a `print()` member that dumps their metadata and state,
useful when inspecting compile-time constants from device code. For debugging
index propagation, start with `none`/`pass_through` morphers to verify plumbing
before layering more complex transformations.

## Further Reading

- [`mint/include/mint/poly/fundamental_morpher_helper.h`](./fundamental_morpher_helper.h)
  – Composition utilities and additional factory helpers.
- [`mint/include/mint/poly/polymorpher_helper.h`](./polymorpher_helper.h)
  – Integration into runtime-polymorphic pipelines.
- [`mint/test/unit_test/poly/test_poly_fundamental_morpher.cu`](../../test/unit_test/poly/test_poly_fundamental_morpher.cu)
  – Unit tests demonstrating typical usage patterns.
