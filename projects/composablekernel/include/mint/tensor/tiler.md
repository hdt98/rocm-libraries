# Mint Tiler

## Concept

The Mint **tiler** pairs a logical iteration space with a compile-time tile
layout so kernels can sweep a tensor in fixed-size windows without recreating
index arithmetic. Given a tensor view, a set of loop dimensions, and a tile
descriptor, the tiler maintains the coordinate state needed to position each
window, propagate it through the tensor's polymorpher, and expose the resulting
byte offset to cooperative load/store routines. Tilers are therefore another
manifestation of Mint's morpher-first architecture: changing the underlying
morpher graph immediately alters how tiles march across memory, translating
logical loop coordinates into physical addresses without any runtime branching.

A tiler instance tracks three concurrent views of the current window:

1. **Loop index** — the current coordinates of the outer iteration space.
2. **Tile-local index** — partition and element coordinates defined by the tile
descriptor.
3. **Tensor coordinate** — the polymorpher-transformed coordinate that targets
global memory alongside a cached byte offset for the window origin.

By maintaining these in lockstep, the tiler lets SIMT tile primitives traverse
complex layouts while remaining backend-neutral.

## Template Parameters and Guarantees

```text
class TensorView,
auto  kLoopDims,
auto  kTileDesc,
auto  kElementTensorDesc,
auto  kPartition
```

The implementation enforces several compile-time contracts:

- `TensorView::ndim() == kTileDesc.top_ndim() + kLoopDims.size()` ensures that
the loop and tile dimensions collectively span the tensor view.
- `kTileDesc.element_ndim() == kElementTensorDesc.top_ndim()` couples the tile
description with the per-participant element descriptor.
- `kElementTensorDesc.bottom_ndim() == 1` requires the element descriptor to map
into a single linear address for efficient byte-offset computation.

These constraints are captured in C++20 `requires` clauses so invalid
configurations fail at compile time.

## Internal State

`mint::tensor::impl::tiler_impl` backs both mutable (`tiler`) and immutable
(`const_tiler`) front-ends. Its key members are:

| Member | Purpose |
|--------|---------|
| `tensor_view_` | Copy of the tensor view that supplies the descriptor and memory access. |
| `coord_` | `coordinate` object storing the current top/bottom indices. |
| `coord_byte_offset_` | Cached byte offset into the tensor's storage. |
| `tile_coord_` | Coordinate inside the tile descriptor (partition + element dimensions). |

During construction the tiler:

1. Zeros all element dimensions within `tile_coord_`.
2. Injects the caller's partition information (e.g., warp or lane identifier).
3. Uses the tile descriptor's polymorpher to propagate indices bottom-up.
4. Merges the supplied loop index with tile-local indices to build the full top
   index and propagates it through the tensor view.
5. Caches the resulting byte offset via `coord_.get_bottom_index()[0] * sizeof(value_type)`.

## Navigation API

| Method | Description |
|--------|-------------|
| `set_tile_index(loop_idx)` | Repositions the tiler to an absolute loop index and recomputes coordinates and byte offset. |
| `move_tiler(loop_delta)` | Increments loop dimensions relative to the current position while keeping tile coordinates fixed. |
| `tile_ndim()` / `loop_ndim()` | Compile-time counts for tile and loop dimensions. |
| `tile_dims()` | Returns the tensor dimensions that belong to the tile (as opposed to the loop). |
| `tile_lengths()` | Compile-time lengths of the tile's top dimensions. |
| `byte_offset()` | Retrieves the cached byte offset for fast memory access. |
| `alias_to_index<alias>()` | Maps a descriptor alias to the current coordinate value. |

All helpers are marked `MINT_HOST_DEVICE` so they can run on host or device
without duplication.

## Mutable vs. Const Tilers

Two thin wrappers expose the implementation:

- `mint::tensor::tiler` — inherits from `tiler_impl` and reports
  `is_const_tiler() == false`. Use this when the underlying tensor view supports
  mutation and kernels need to write through the view.
- `mint::tensor::const_tiler` — inherits the same implementation but marks itself
  read-only. When the tensor view advertises `is_const_tensor_view()`, the
  factory automatically returns this variant to prevent accidental writes.

Both wrappers forward every constructor and member function provided by
`tiler_impl`.

## Factory Helpers

Two overloads of `make_tiler` select the correct wrapper based on the constness
of the supplied tensor view:

```text
make_tiler(view,
           integral_constant<nd_index<kLoopNDim>, kLoopDims>,
           loop_idx,
           distributed_tensor<kTileDesc, kElementTensorDesc, Memory>,
           constant<kPartition>)
```

Arguments include the loop dimensions, the current loop index, the distributed
tensor type expected by downstream kernels, and the partition descriptor. The
factory copies the loop index into the new tiler and performs the initialization
sequence described above.

## Integration Workflow

1. **Describe the tile.** Build `kTileDesc` and `kElementTensorDesc` using the
   polymorpher tooling shared with distributed tensors. Partition metadata should
   reflect which lanes or warps collaborate on the tile.
2. **Select loop dimensions.** Choose which tensor dimensions advance via outer
   loops (e.g., tile grid indices). Remaining dimensions become the tile's
   spatial axes.
3. **Instantiate a tiler.** Call `make_tiler` with a tensor view and the current
   loop index inside your kernel or host routine.
4. **Advance through tiles.** Use `set_tile_index` for absolute positioning or
   `move_tiler` for relative iteration. Each step exposes the cached byte offset
   and tensor coordinate for load/store helpers.
5. **Cooperate with distributed tensors.** Tile primitives such as
   `load_no_shuffle_tiler` or `store_no_shuffle_tiler` consume the tiler together
   with a pre-allocated `distributed_tensor`, staging data into registers or
   shared memory according to the descriptors.

## Relationship to Other Components

- **Tensor Views:** Provide the polymorpher-driven logical-to-physical mapping.
  The tiler relies on the view's descriptor when propagating indices.
- **Distributed Tensors:** Receive the tile fragments. Their descriptors must
  align with `kTileDesc` and `kElementTensorDesc` so cooperative loads land in the
  correct registers or shared memory slots.
- **Morphers:** Tile and tensor descriptors compose fundamental morphers
  (pass-through, split/merge, Z2 transforms) to describe swizzled or reshaped
  layouts. The tiler is agnostic to the specific morphers as long as the
  descriptor contracts hold.
- **SIMT Tile Kernels:** Higher-level kernels in `mint/tile/generic` and
  backend-specialized directories use the tiler to step through global memory
  while masking inactive lanes and handling boundary conditions.

## Practical Guidance

- **Cache-friendly iteration:** Prefer `move_tiler` when sweeping tiles in a
  nested loop; it reuses coordinate state and only updates loop dimensions.
- **Partition awareness:** Ensure `kPartition` matches the cooperative entity
  executing the kernel (lane, warp, block). Mismatches lead to incorrect
  distribution of tile fragments.
- **Debugging:** Inspect the tensor view's descriptors or the tiler's `coord_`
  and `tile_coord_` within device-side debuggers to verify loop propagation.
- **Const correctness:** Allow the factory to choose between `tiler` and
  `const_tiler` to guard write paths when staging read-only operands.

## References

- [`mint/include/mint/tensor/tiler.h`](./tiler.h)
- [`mint/include/mint/tensor/distributed_tensor.md`](./distributed_tensor.md)
- [`mint/include/mint/tensor/tensor_view.md`](./tensor_view.md)
- [`mint/include/mint/tile/generic/load_no_shuffle_tiler.h`](../tile/generic/load_no_shuffle_tiler.h)
- [`mint/include/mint/tile/generic/store_no_shuffle_tiler.h`](../tile/generic/store_no_shuffle_tiler.h)
- [`mint/test/unit_test/tile/simt/test_tile_simt_warp_load_store_z2_distribution_tiler_generic_layout.cu`](../../../../test/unit_test/tile/simt/test_tile_simt_warp_load_store_z2_distribution_tiler_generic_layout.cu)
