# Task 02: Distribution Configuration

## Overview

This example demonstrates the unified distribution API that creates
thread-to-tile mappings for block-level data copy operations in GEMM kernels.

A distribution defines how threads in a workgroup collaboratively cover a 2D tile:
- **Block copy A distribution**: Maps BlockSize threads to an MPerBlock x KPerBlock tile
- **Block copy B distribution**: Maps BlockSize threads to a KPerBlock x NPerBlock tile

## Backend Mapping

| Unified API | CK_TILE | MINT |
|-------------|---------|------|
| `make_block_copy_a_distribution<BS, M, K, Vec>()` | `tile_distribution_encoding_pattern_2d<BS, M, K, Vec, thread_raked>::make_2d_static_tile_distribution()` | `distributed_tensor_descriptor` via morphers: merge(P), split(M), split(K) |
| `make_block_copy_b_distribution<BS, N, K, Vec>()` | `tile_distribution_encoding_pattern_2d<BS, K, N, Vec, thread_raked>::make_2d_static_tile_distribution()` | `distributed_tensor_descriptor` via morphers: merge(P), split(N), split(K) |
| `get_elements_per_thread(dstr)` | `dstr.get_ys_to_d_descriptor().get_element_space_size()` | `Distribution::element_size()` |
| `get_num_tile_dims(dstr)` | `Distribution::get_num_of_dimension_x()` | `Distribution::top_ndim()` |

## Key Invariant

For any valid parameters, `elements_per_thread = TileDim0 * TileDim1 / BlockSize`.
Both backends produce distributions that satisfy this invariant.

## Thread Decomposition (MINT model)

For A distribution with BlockSize=256, M=128, K=64, KPack=8:
```
K0 = K/KPack = 8         tid_k = K0 = 8
tid_m = BS/tid_k = 32    M0 = M/tid_m = 4
Thread ID = merge(tid_m, tid_k)
Element dims: M_0 (=4 elements), K_1 (=8 elements)
Per-thread: 4 * 8 = 32 elements
```

## Build & Run

```bash
# CK_TILE backend
ninja example_unified_tile_02_distribution_ck
./bin/example_unified_tile_02_distribution_ck

# MINT backend
ninja example_unified_tile_02_distribution_mint
./bin/example_unified_tile_02_distribution_mint
```
