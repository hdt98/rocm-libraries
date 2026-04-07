# Task 05: Store Operations

Unified store tile operations across CK_TILE and MINT backends.

## What This Does

Stores data from VGPR (distributed tensor) through a tile window back to
global or shared memory. This is the write counterpart to Task 04 (Load).

### Pipeline: load -> **store** -> (next iteration)

```
1. Load: DRAM -> VGPR (via load_tile)
2. Store: VGPR -> LDS or DRAM (via store_tile)
3. Verify: check stored data matches original
```

## Unified API

```cpp
// Store distributed tensor through a tile window
ops::store_tile(dst_window, tile);
```

## Backend Mapping

| Unified | CK_TILE | MINT |
|---------|---------|------|
| `store_tile(win, tile)` | `ck_tile::store_tile(win, tile)` | `masked_store_no_shuffle_vectorized<vec, len, freeze>(win, mask, tile)` |

## Key Design Notes

- MINT's `masked_store_no_shuffle_vectorized` requires vector_dims, vector_lengths,
  and freeze_dims - all auto-derived from the window's distribution (same pattern
  as load.hpp).
- The test verifies data integrity: load from src, store to dst, then check
  that dst[i] == src[i] at multiple positions on the device.

## Build & Run

```bash
ninja example_unified_tile_05_store_ck
./bin/example_unified_tile_05_store_ck

ninja example_unified_tile_05_store_mint
./bin/example_unified_tile_05_store_mint
```
