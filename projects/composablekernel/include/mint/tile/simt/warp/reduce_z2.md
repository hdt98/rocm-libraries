# Warp-Level Z2 Reduction (`reduce_z2`)

## Purpose

`mint::tile::simt::warp::reduce_z2` provides warp-scope reductions when the
source tensor is described by Z<sup>2</sup> (Morton/Z-order) polymorphers. Z<sup>2</sup>
layouts interleave logical dimensions into bit lanes, which means threads in a
warp may hold values that must be combined across both *partition* (P) and
*element* (E) dimensions. The `reduce_z2` helpers bridge this layout to a warp
shuffle friendly form so that the reduction can happen without materializing a
layout-specific kernel.

## Key Ideas

1. **Bottom-dimension extraction** – The helper splits the source Z<sup>2</sup>
morpher into two new morphers: one that covers the P dimensions being reduced
and one that covers the E dimensions that remain.
2. **Dimension fusion** – P and E morphers are fused into a temporary descriptor
that exposes a linear lane ordering for the reduced outputs.
3. **Two phase reduction** – Each thread first accumulates its local elements and
then participates in SRAM/register shuffle stages that match the Z<sup>2</sup>
bit pattern. The shuffle stages are driven by the top-down matrix encoded by the
polymorpher so that only threads that share a partition bit communicate.
4. **Descriptor reconstruction** – The helpers return or write into distributed
tensors that maintain the original partition metadata even though the reduced
dimension has collapsed to length 1 on the top (logical) axis.

## API Surface

`reduce_z2` is overloaded for progressively higher-level use cases:

- **Core primitive**
  ```
  reduce_z2(p_src, SrcDstr{}, bot_e_dims, bot_p_dims, reduced_dim,
           SrcLayout{}, f_reduce);
  ```
  Returns a tuple containing the temporary distribution, layout, and VGPR-owned
  scratch tile populated with the reduction results. This overload is useful when
  the caller wants to control how the reduced data is consumed.

- **In-place shuffle into destination**
  ```
  reduce_z2(p_dst, p_src, SrcDstr{}, DstDstr{}, bot_e_dims, bot_p_dims,
           reduced_dim, SrcLayout{}, DstLayout{}, f_reduce);
  ```
  Performs the reduction and then calls `shuffle_z2` to place the results into an
  output distribution.

- **Distributed-tensor convenience wrappers**
  ```
  reduce_z2(dst_dstr_tensor, src_dstr_tensor, reduced_dim, f_reduce);
  auto reduced_tensor = reduce_z2(src_dstr_tensor, reduced_dim, f_reduce);
  ```
  These work directly with `distributed_tensor` objects, reusing their
  polymorphers to supply the template arguments used by the lower-level
  overloads.

All overloads accept an `FReduce` functor/lambda that must be callable as
`float(float, float)` (or the appropriate numeric type) and implement the binary
reduction operator, such as `std::plus<float>{}` or a custom min/max.

## Preconditions

- The source descriptors (`SrcDstr`/`SrcLayout`) must be Z<sup>2</sup> linear
  morphers (`poly::is_z2_linear_morpher`).
- The selected P and E dimensions must satisfy `impl::is_valid`: each top-row of
  the Z<sup>2</sup> matrix must depend on **either** E or P bottom bits but not
  both. This guarantees that partition stages route data between the correct
  lanes.
- Warp size must match the encoded partition bit width (enforced via static
  assertions against `MINT_WARP_SIZE`).

## Execution Flow

1. **Morpher extraction**
   - `extract_by_bottom_dims` slices the original polymorpher into P and E
     morphers that only expose the subset of bottom dimensions involved in the
     reduction.
2. **Top-dimension reduction**
   - `make_reduced_p` and `make_reduced_e` rewrite the top dimension lengths so
     the reduced dimension has length 1 while keeping ordering information in the
     transformation matrix.
3. **Fusion and layout synthesis**
   - `fuse_bottom_dims` interleaves bits from the P and E morphers to build a
     temporary descriptor whose bottom space matches a warp-friendly linear
     offset.
4. **Intra-thread accumulation**
   - Threads iterate local E coordinates and accumulate into VGPR-backed scratch
     (`owned_vgpr_memory`).
5. **Inter-thread accumulation**
   - The binary matrix produced by the polymorpher determines which lane bits are
     toggled per stage. The helper calls `__shfl_xor_sync`/`__shfl_xor` with these
     masks to combine values across lanes, applying `f_reduce` at each stage.
6. **Result handoff**
   - The low-level overload returns the tuple `(tmp_dstr, tmp_layout, scratch)`,
     while the higher-level overloads either shuffle the results into a user
     provided destination tensor or materialize a new distributed tensor with the
     reduced dimension collapsed.

## Tips for Use

- Prefer the distributed-tensor overloads unless you need direct access to the
  temporary `owned_vgpr_memory` buffer.
- When chaining multiple reductions, reuse the same `FReduce` functor to avoid
  code duplication and ensure consistent associativity/commutativity.
- Unit tests demonstrating the API live in:
  - `mint/test/unit_test/tile/rocm/test_tile_rocm_z2_reduce.hip`
  - `mint/test/unit_test/tile/rocm/test_tile_rocm_raw_z2_reduce.hip`

## Relationship to `shuffle_z2`

`reduce_z2` and `shuffle_z2` are complementary. The reduction helper produces a
packed intermediate buffer whose layout may not match the caller's target Z<sup>2</sup>
partitioning. The overload that accepts both source and destination descriptors
invokes `shuffle_z2` to remap from the fused temporary descriptor back into the
requested destination layout.
