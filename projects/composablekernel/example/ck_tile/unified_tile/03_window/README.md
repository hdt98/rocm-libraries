# Task 03: Tile Window Abstraction

Unified tile window creation and movement across CK_TILE and MINT backends.

## What This Does

A tile window combines a tensor view with a distribution to create a
thread-mapped view into GPU memory. Each thread in the workgroup "sees"
its own portion of the tile through the window.

### Pipeline: descriptor -> view -> distribution -> **window**

```
1. Descriptor: define tensor shape (M x K)
2. View: attach memory pointer to descriptor
3. Distribution: define how threads decompose the tile
4. Window: combine view + distribution + origin offset
5. Move: slide the window along K for the next iteration
```

## Unified API

```cpp
// Create window from view + distribution
auto win = window::make_tile_window(view, tile_lengths, origin, distribution);

// Move window along a dimension
window::move_window(win, step);
```

## Backend Mapping

| Unified | CK_TILE | MINT |
|---------|---------|------|
| `make_tile_window(view, lens, origin, dstr)` | `ck_tile::make_tile_window(view, lens, origin, dstr)` | `mint::make_distributed_window(view, origin, constant<dstr>, constant<partition>)` |
| `move_window(win, step)` | `win.move(step)` | `mint::tensor::move_window(win, step)` |

## Key Design Notes

- MINT requires an explicit `partition` object (`thread_in_this_block<BS>`) and
  an `element_layout` descriptor. Both are auto-generated internally from the
  distribution's `partition_size()` and `element_lengths()`.
- CK_TILE handles partitioning internally via lane_id/warp_id from the distribution.
- The `lengths` parameter is used by CK_TILE but ignored by MINT (MINT infers
  tile size from the distribution's `top_lengths()`).

## Build & Run

```bash
# CK_Tile backend
ninja example_unified_tile_03_window_ck
./bin/example_unified_tile_03_window_ck

# MINT backend
ninja example_unified_tile_03_window_mint
./bin/example_unified_tile_03_window_mint
```
