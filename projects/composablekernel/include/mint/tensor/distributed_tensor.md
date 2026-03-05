# Mint Distributed Tensors

## Concept

A distributed tensor combines partition and element dimensions so a cooperating
set of threads, warps, or devices collectively holds a tile of a larger tensor
in private memory. Partition dimensions decide which participant owns each
chunk, while element dimensions specify the per-owner fragment that lives in
VGPRs, shared memory, or another private store. Instead of every lane fetching
values independently, the group first pulls the tile into private memory,
performs the fused math step, then writes the result back. The layout of that
scratchpad is defined entirely at compile time through polymorphers.

Morpher-driven descriptors are therefore the core of Mint: every distributed
tensor is just a concrete manifestation of the morpher graph that routes logical
indices to physical storage. Changing the morpher changes the distributed tensor
without touching runtime code, which is the foundational design goal across the
library.

Two sets of dimensions describe every distributed tensor:

- **Partition dimensions** say *who* owns each chunk. They map logical indices
  to specific threads, warps, or blocks.
- **Element dimensions** say *what* each owner stores locally. They describe the
  per-thread fragment (strides, vector length, packing order) that lives in
  private memory.

The combination lets Mint express sophisticated tilings—blocking, swizzling,
vectorization—while remaining agnostic to hardware backend.

## Components

A `mint::tensor::distributed_tensor` instance ties together three pieces:

1. `kDstrTensorDesc` – the **distribution descriptor** that captures partition
   and element dimensions along with the polymorpher that maps logical indices to
   physical offsets.
2. `kElementTensorDesc` – the **element descriptor** that focuses on the shape of
   the per-thread fragment (vector widths, packed strides, etc.).
3. `Memory` – a **memory view** representing the actual storage (owned VGPR
   registers, shared-memory slab, or pointer into global memory).

Partition dimensions explain how the tile is split across cooperative entities;
element dimensions explain the local payload each entity holds. Together they
allow tile routines to load, compute, and store with zero runtime bookkeeping.

## Typical Workflow

1. **Describe the layout** with polymorphers (fundamental + Z2 helpers) and build
   a distributed tensor descriptor via helpers such as
   `make_distributed_tensor_vgpr_z2` or `make_simple_distributed_tensor_vgpr_z2`.
2. **Materialize storage** in the desired address space (VGPR, shared, global).
3. **Construct the distributed tensor** using one of the `make_distributed_tensor`
   factories.
4. **Invoke tile kernels** (`mint/tile/simt/*`) to cooperatively load from global
   memory, compute, and store results, all using the descriptor’s index
   propagation.

## Key API Surface

| Member / Helper | Purpose |
|-----------------|---------|
| `top_lengths()` | Returns the logical extent represented by the tile. |
| `dstr_tensor_desc()` / `element_tensor_desc()` | Expose the compile-time descriptors (partition + element metadata). |
| `memory()` | Accessor for the underlying memory view (mutable/const). |
| `fill(value)` | Broadcast-initializes the cooperative scratchpad. |
| `element<idx>` / `element_vector<idx, N>` | Compile-time element access for scalar or vector fragments. |
| `sharded_element<sequence>` | Maps per-thread shard coordinates to element coordinates. |
| `reshape_logical`, `reorder_logical`, `swizzle_logical` | Rebuild the descriptor with a transformed polymorpher while reusing storage (single-morpher constraint enforced). |
| `reshape_element`, `reshape_partition` | Adjust element or partition dimensions without reallocation. |

## Transform APIs

Distributed tensors expose two categories of transforms that operate entirely at
compile time. Logical transforms—`reshape_logical`, `reorder_logical`, and
`swizzle_logical`—rebuild the distribution descriptor with an alternate
polymorpher. Because the storage view is preserved, these calls are ideal for
trying new tile layouts or swizzles while keeping the same cooperative scratch
allocation. Element and partition transforms—`reshape_element` and
`reshape_partition`—tune the per-owner fragment or ownership mapping without
triggering a reallocation. Together these APIs let authors iterate on tilings and
fragment shapes rapidly while guaranteeing that the resulting distributed tensor
remains compatible with existing tile kernels.

All access ultimately passes through the polymorpher in `kDstrTensorDesc`, so any
compile-time reshape or swizzle executed there is automatically honored by the
tile APIs.

## Integration with Tensor Views and Tiles

Distributed tensors usually originate from tensor views:

1. A `tensor_view` provides logical indexing over global memory via a polymorpher.
2. The view is converted into a distributed tensor to stage data in private
   memory cooperatively.
3. Tile primitives (`load_vectorized_freezed_dims`, `matmul_mn_mk_kn_no_shuffle`,
   etc.) consume the distributed tensor for actual computation.

This pipeline keeps layout logic centralized in polymorphers while allowing tile
kernels to remain simple and backend-neutral.

## Practical Notes

- **Choose the right memory view**: use VGPR-backed helpers for register tiles,
  or shared-memory/pointer views when coordinating across blocks.
- **Leverage descriptors**: pass `dstr_tensor_desc()` and
  `element_tensor_desc()` into other helpers to avoid recomputing layouts.
- **Debugging**: `print()` emits descriptor and memory information, which is
  invaluable when validating new morphers or tile schedules.
- **Backend portability**: because the descriptors are purely compile-time, the
  same distributed tensor definition can feed SIMT-generic, ROCm, or future
  backend-specific tile kernels.

## References

- [`mint/include/mint/tensor/distributed_tensor.h`](./distributed_tensor.h)
- [`mint/include/mint/tensor/tensor_view.md`](./tensor_view.md)
- [`mint/include/mint/tensor/tiler.md`](./tiler.md)
- [`mint/include/mint/tile/tile_overview.md`](../tile/tile_overview.md)
- [`mint/include/mint/poly/polymorpher_doc.md`](../poly/polymorpher_doc.md)
