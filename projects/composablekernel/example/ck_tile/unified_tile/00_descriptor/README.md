# Task 00: Unified Tensor Descriptor

## Overview

This example demonstrates the unified tensor descriptor API that works across
both CK_TILE and MINT backends. The descriptor is the foundation of the tile
programming model - it maps logical multi-dimensional indices to physical
memory offsets.

## What This Task Covers

- Creating packed (row-major) tensor descriptors via `make_descriptor(M, K)`
- Creating descriptors with explicit strides via `make_descriptor_with_strides()`
- Querying descriptor properties: number of dimensions, lengths

## API Summary

| Unified API | CK_TILE | MINT |
|-------------|---------|------|
| `make_descriptor(M, K)` | `make_naive_tensor_descriptor_packed(make_tuple(M, K))` | `make_aliased_naive_packed_tensor_descriptor(aliases, offset, {M, K})` |
| `make_naive_tensor_descriptor(lens, strides)` | `make_naive_tensor_descriptor(tuple, tuple)` | `make_aliased_naive_tensor_descriptor(aliases, offset, nd_index, nd_index)` |

## Build

```bash
cd /root/workspace/build

# CK_TILE backend
ninja example_unified_tile_00_descriptor_ck

# MINT backend
ninja example_unified_tile_00_descriptor_mint
```

## Run

```bash
./bin/example_unified_tile_00_descriptor_ck
./bin/example_unified_tile_00_descriptor_mint
```

## Expected Output

```
[Unified Tile] Descriptor Example (Backend: CK_Tile)
PASSED: 2D packed descriptor created
PASSED: 3D packed descriptor created
All tests passed!
```
