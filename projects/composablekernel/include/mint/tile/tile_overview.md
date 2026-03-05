# Mint Tile Module Overview

## Purpose

The `mint::tile` module connects polymorpher-driven tensor layouts to hardware
execution primitives. Because Mint is architected around morphers, every tile
operation is a direct consumer of the morpher graph that shaped its tensors.
Those morphers provide the canonical mapping from logical coordinates to
physical storage, and tile kernels simply walk that mapping in a cooperative
fashion. The module provides building blocks for moving tiles of data between
memory spaces (global, shared, VGPR), partitioning work across threads or warps,
and issuing backend-optimized matrix multiply, reduction, and vectorized
load/store operations.

## Directory Map

| Subdirectory | Focus | Highlights |
|--------------|-------|------------|
| `simt/` | Backend-agnostic SIMT layer | Warp/block load/store, partition helpers, wrapper matmul APIs |
| `generic/` | Backend-independent implementations | Core "no-shuffle" primitives for load/store/matmul/reduce/elementwise/atomic |
| `rocm/` | AMD GPU specialization | MFMA/XDL warp matmul, async copy, AMD buffer load |

## SIMT Layer

The SIMT layer exposes the user-facing tile APIs. It expects distributed tensor
descriptors generated from polymorphers and coordinates with thread partitions.
All functions are header-only templates usable from CUDA, ROCm, or other SIMT
platforms.

- `simt/load.h`, `simt/store.h`: `load_vectorized_freezed_dims`,
  `store_vectorized_freezed_dims` move tiles between distributed tensors and
  VGPR/shared memory with optional masking.
- `simt/warp/matmul.h`: Warp-level matmul wrapper that delegates to the generic
  kernel (`generic/matmul_no_shuffle.h`).
- `simt/partition.h`: Concepts like `thread_in_this_warp`,
  `thread_in_this_block` that provide per-thread coordinates for sharded tiles.
- `simt/block/load.h`, `simt/block/store.h`: Block-scoped masked load/store that
  compose the lower-level generic primitives.
- `simt/warp/reduce_z2.h`, `simt/warp/shuffle_z2.h`: Support for reductions and
shuffles when morphers introduce Z-order/permutation layouts. See
[`simt/warp/reduce_z2.md`](simt/warp/reduce_z2.md) for a walkthrough of the
warp-level Z^2 reduction helper and
[`simt/warp/shuffle_z2.md`](simt/warp/shuffle_z2.md) for the corresponding
warp-level shuffle.

## Generic Core

The generic layer implements the backend-neutral algorithms that the SIMT layer
wraps. They operate purely on distributed tensor descriptors and memory views,
leaving hardware specialization to upper layers or backend directories.

- `generic/matmul_no_shuffle.h`: `matmul_mn_mk_kn_no_shuffle` performs the core
  nested-loop matrix multiply using polymorpher-friendly tensor views without
  requiring intra-warp shuffles.
- `generic/load_no_shuffle*.h`, `generic/store_no_shuffle*.h`: Families of
  masked load/store helpers, including vectorized and byte-offset variants.
- `generic/reduce_no_shuffle.h`, `generic/elementwise_no_shuffle.h`,
  `generic/atomic_add_no_shuffle.h`: Reduction, element-wise arithmetic, and
  atomic updates over tiles.
- `generic/*_tiler.h`: Iteration utilities that integrate with the polymorpher
  tiler abstractions.

## Backend Specializations

### ROCm (`tile/rocm`)
- `warp/matmul_xdl.h`: AMD MFMA/XDL matmul intrinsics (e.g.,
  `matmul_xdl<kMPerFMA, kNPerFMA>`).
- `warp/matmul_dl.h`: Data-layout variant of warp matmul.
- `load_no_shuffle_vectorized_async_copy.h`, `amd_buffer_load.h`: High-throughput
  vectorized load paths leveraging ROCm builtins and async copy queues.

## Using Tile APIs with Morphers

1. **Describe Layouts**: Build distributed tensor descriptors (`kDstrTensorDesc`,
   `kElementTensorDesc`) from polymorphers that encode how logical indices map to
   physical storage.
2. **Instantiate Distributed Tensors**: Convert tensor views into
   `distributed_tensor` objects backed by VGPR/shared/global memory.
3. **Construct a Tiler**: Use `make_tiler` to pair the loop space with the tile
   descriptor so cooperative loads/stores can march across the tensor view.
4. **Issue Tile Operations**: Call SIMT/generic load, store, matmul, or reduce
   functions with the distributed tensors, partition descriptors, and tiler.
5. **Backend Dispatch**: For ROCm targets, include the corresponding headers to
   leverage specialized kernels.

Because tile APIs consume polymorpher-driven descriptors, they work seamlessly
with both fundamental and Z2 morphers. Morphers define the index propagation;
tile utilities simply execute the resulting memory movement or compute pattern.

## References

- [`mint/include/mint/tensor/distributed_tensor.h`](../tensor/distributed_tensor.h)
  – Distributed tensor descriptors consumed by tile APIs.
- [`mint/include/mint/poly/polymorpher_doc.md`](../poly/polymorpher_doc.md)
  – Background on building polymorpher stacks for tiles.
- [`mint/test/kernel`](../../test/kernel) – Kernel examples combining tensor
  descriptors with tile operations (e.g., ROCm GEMM tests).
