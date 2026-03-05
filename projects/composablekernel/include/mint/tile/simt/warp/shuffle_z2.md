# Warp-Level Z2 Shuffle (`shuffle_z2`)

## Purpose

`mint::tile::simt::warp::shuffle_z2` remaps data that lives in Z<sup>2</sup>
(Morton/Z-order) distributions across the lanes of a warp without touching
global memory. When partitions or elements are reshaped by polymorphers, lane
IDs no longer match contiguous indices. The shuffle helper interprets the
underlying Z<sup>2</sup> matrices and issues the appropriate warp shuffle
instructions so that data arrives at the destination descriptor with the correct
bit interleaving.

## Key Ideas

1. **Bit-level view of bottom dimensions** – The helper enumerates the bit ranges
   (`bot_bit_begins_`/`bot_bit_ends_`) that correspond to the element (E) and
   partition (P) dimensions selected by the caller.
2. **Matrix factorization** – By chaining `invert(DstDstr)` with `SrcDstr` we
   obtain a morpher whose matrix expresses `{e_dst, p_dst} -> {e_src, p_src}` in
   Z<sup>2</sup> space. Each phase of the algorithm slices this matrix to isolate
   E and P contributions.
3. **Two-stage exchange** – Threads first shuffle elements within their own lane
   (intra-thread) using the element layout morphers, then optionally exchange
   data across lanes (inter-thread) when the P matrices differ.
4. **Backend abstraction** – The template uses `__shfl_xor_sync`/`__shfl_xor`
   intrinsics underneath; compile-time checks on `MINT_WARP_SIZE` ensure the
   encoded bit width matches the backend’s lane count.

## API Surface

`shuffle_z2` mirrors the reduction helper’s layering to support different usage
sites:

- **Core primitive**
  ```
  shuffle_z2(p_dst, p_src, SrcDstr{}, DstDstr{},
            constant<kEDims>{}, constant<kPDims>{},
            SrcElmLayout{}, DstElmLayout{});
  ```
  Copies data from `p_src` to `p_dst`, reinterpreting the Z<sup>2</sup>
  polymorphers while respecting the selected element and partition dimensions.

- **Distributed tensor overload**
  ```
  shuffle_z2(dst_dstr_tensor, src_dstr_tensor);
  ```
  Convenience wrapper for tensors whose descriptors use a single morpher and
  share element/partition dimensions.

The function assumes the element layouts expose a single morpher (packed
VGPR/shared tile) so that their `kBottomLengths` match the number of values each
lane owns.

## Preconditions

- Source and destination morphers must be Z<sup>2</sup> linear (enforced by the
  descriptor types themselves).
- `SrcDstr::kBottomLengths == DstDstr::kBottomLengths` – data volume per lane is
  unchanged; only ordering differs.
- Element/partition dimensions passed via `kEDims`/`kPDims` must align between
  source and destination descriptors. The distributed tensor overload checks this
  via the `compatible_shuffle_dimensions` concept.

## Execution Flow

1. **Bit extraction** – `get_bot_bits` walks the bottom dimension metadata to
   build the list of bit indices belonging to the selected dimensions. These
   indices feed `z2_matrix_extract_columns` to isolate the relevant columns of
   the top-down matrices.
2. **Matrix chaining** – `dst2src_shuffle_m = chain(invert(DstDstr{}), SrcDstr{})`
   provides a composite morpher describing how destination coordinates map back
   into the source lane/element space.
3. **Morpher slicing** – `extract` obtains separate morphers for elements and
   partitions. The element morpher is chained with `SrcElmLayout` so that each
   lane can compute the local offsets it needs to gather from the source tile.
4. **Intra-thread move** – Each thread reads its `kEPerThread` values directly
   from `p_src` and writes them to the corresponding position in `p_dst` using the
   element morphers.
5. **Inter-thread shuffle (conditional)** – If source and destination P matrices
   are identical, no cross-lane shuffle is necessary. Otherwise the partition
   morpher maps the destination lane bitset into the source lane ID, and
   `__shfl` pulls values from that lane for every element index.

## Tips for Use

- Pair `shuffle_z2` with `reduce_z2` when a reduction produces a scratch buffer
  whose fused descriptor must be realigned to a user-facing distribution.
- If you only need intra-thread reordering (identical partition morphers), the
  helper detects this at compile time and avoids emitting shuffle instructions.
- Unit tests illustrating usage live in:
  - `mint/test/unit_test/tile/rocm/test_tile_rocm_z2_shuffle.hip`
  - `mint/test/unit_test/tile/rocm/test_tile_rocm_raw_z2_shuffle.hip`

## Relationship to `reduce_z2`

`reduce_z2` fuses partition and element morphers to perform warp reductions in a
scratch tile. The overload that accepts destination descriptors subsequently
calls `shuffle_z2` to restore the requested Z<sup>2</sup> ordering. Reading both
helpers together clarifies how reductions and layout permutations cooperate at
warp scope.
