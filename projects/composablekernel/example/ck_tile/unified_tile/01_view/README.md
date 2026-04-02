# Task 01: Unified Tensor View

## Overview

This example demonstrates the unified tensor view API that works across both
CK_TILE and MINT backends. A tensor view combines a memory pointer with a
descriptor, providing a structured way to access multi-dimensional data.

## What This Task Covers

- Creating global memory tensor views via `make_tensor_view<global>(ptr, desc)`
- Creating views from pointer + lengths + strides (CK_TILE)
- Padding views via `pad_view()` (CK_TILE pads, MINT returns unchanged)
- Reading data through the view on device

## API Summary

| Unified API | CK_TILE | MINT |
|-------------|---------|------|
| `make_tensor_view<global>(ptr, desc)` | `ck_tile::make_tensor_view<global>(ptr, desc)` | `mint::make_tensor_view(desc, make_global_memory_view(ptr, size))` |
| `make_tensor_view<shared>(ptr, desc)` | `ck_tile::make_tensor_view<lds>(ptr, desc)` | `mint::make_tensor_view(desc, make_shared_memory_view(ptr, size))` |
| `pad_view(view, tile_lens, do_pads)` | `ck_tile::pad_tensor_view(...)` | returns view unchanged |

## Build

```bash
# Inside Docker container
cd /root/workspace/build

# CK_TILE backend
ninja example_unified_tile_01_view_ck

# MINT backend
ninja example_unified_tile_01_view_mint
```

## Run

```bash
./bin/example_unified_tile_01_view_ck
./bin/example_unified_tile_01_view_mint
```

## Expected Output

```
[Unified Tile] Tensor View Example (Backend: CK_Tile)
PASSED: Global tensor view created and data accessible
All tests passed!
```
