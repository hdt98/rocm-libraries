# Design Overview: Mint Z2 Morphers

Mint's Z2 morphers, inspired by Triton’s linear layout (https://arxiv.org/html/2505.23819v1), treat every tensor-layout transformation as a linear map over the binary field. Because they materialize lane interactions as explicit GF(2) matrices, warp-level reductions and shuffles fall out as straight-line sequences of GPU shuffle instructions—each matrix row names the partner lane to read via `__shfl_*` or `ds_*` intrinsics. This document explains how the design is organized, which compile-time artifacts it produces, and how those pieces compose with the broader polymorpher system.

## Design Objectives

- **Bit-level control.** Encode logical-to-physical index flow explicitly as GF(2) matrices so that every bit movement is inspectable and verifiable.
- **Composable operators.** Provide a closed algebra of helper constructors that assemble complex layouts from identity, permutation, swizzle, and fusion primitives.
- **Static guarantees.** Resolve rank checks, inverses, and shuffle schedules at compile time so generated kernels can remain branchless and deterministic.

## Binary Vector Spaces

Logical tensor dimensions `L₀ … L_{T−1}` contribute `⌈log₂ L_t⌉` digits apiece, forming the domain vector space `V_top ≅ GF(2)^m`. Physical tiling dimensions `B₀ … B_{B−1}` do the same for the codomain `V_bottom ≅ GF(2)^n`. Metadata constants (`kTopLengths`, `kBottomLengths`) capture the ordering of these digits so we can reason about contiguous bit slices per dimension.

## Matrix-Centric Morphers

Each morpher stores a compile-time matrix `M = kTopDownZ2Matrix ∈ GF(2)^{n×m}` that realizes the linear map `Φ(x) = Mx`. Because indices are packed bitsets, applying the morpher amounts to masking the relevant bits, xoring them together, and reassembling the physical index. This matrix is the canonical representation for every downstream operation:

- **Forward propagation.** `propagate_index_and_delta_top_down` multiplies both indices and differentials by `M`, mirroring the tangent-vector calculus from the GF(2) layout literature.
- **Invertibility.** `can_bottom_up_` performs compile-time Gaussian elimination on `M` to test rank, emit inverses when available, and characterize fibers when it is not square.

## Constructor Algebra

`mint/include/mint/poly/z2_linear_morpher_helper.h` implements a minimal generator set that developers compose to build `M`:

| Helper | Matrix Operation | Design Intent |
|--------|------------------|---------------|
| `make_z2_pass_through_morpher` | Identity | Start from base logical layout |
| `reorder_*` | Permutation matrix | Relabel coordinates |
| `swizzle_*` | Elementary row/column operations | Implement XOR shear stages |
| `reshape_*` | Change-of-basis blocks | Refine bit slices without altering total rank |
| `concat(lhs, rhs)` | Block diagonal | Direct-sum independent sublayouts |
| `chain(lhs, rhs)` | Matrix product | Sequential composition of operators |
| `extract` | Submatrix selection | Project to subspaces |
| `fuse` | Block matrix with shared faces | Glue layouts across shared coordinates |

The helpers enforce invariants (matching bit counts, compatible slices, coherent boundaries) so the generated matrix always respects the underlying vector-space model.

## Authoring Workflow

1. **Model logical geometry.** Choose `kTopLengths` for the input tensor dimensions.
2. **Select physical tiling.** Decide on `kBottomLengths` that represent the warp/block/memory axes desired on hardware.
3. **Compose the matrix.** Apply helper constructors to obtain the GF(2) matrix that matches the intended layout.
4. **Validate structure.** Use `is_z2_matrix_invertible` or related utilities to confirm rank properties or to reason about aliasing fibers.
5. **Integrate into polymorphers.** Treat the morpher as a chart inside Mint's polymorpher atlas; it can be chained with fundamental morphers or other linear pieces.

## Complementing Fundamental Morphers

Fundamental morphers in Mint implement integer arithmetic (splits, merges, shifts) that lie outside the strict linear category. Z2 morphers cover the XOR-linear slice (`GL(V_top, GF(2))` and relatives). In practice they are often chained: fundamental morphers shape high-level tiling factors, while Z2 morphers fine-tune bit-level permutations and swizzles with algebraic guarantees.

## Key Implementation Artifacts

- [`mint/include/mint/poly/z2_linear_morpher.h`](./z2_linear_morpher.h) — definition of the template, compile-time invariants, and propagation routines.
- [`mint/include/mint/poly/z2_linear_morpher_helper.h`](./z2_linear_morpher_helper.h) — helper constructors that assemble matrices.
- [`mint/include/mint/poly/z2.h`](./z2.h) — Gaussian elimination, rank checks, and inverse construction over GF(2).
- [`mint/test/unit_test/poly/test_poly_z2_linear_morpher.cu`](../../test/unit_test/poly/test_poly_z2_linear_morpher.cu) — reference kernels demonstrating concrete matrices and warp-collective synthesis.

By designing morphers around explicit GF(2) matrices, Mint guarantees that bit-level layouts, warp collectives, and differential reasoning all share a single algebraic vocabulary.
