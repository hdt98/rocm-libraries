# Task 04: Load Operations

Unified tile load operation across CK_TILE and MINT backends.

## What This Does

Loads data from global/shared memory through a tile window into per-thread
registers (VGPR). Each thread in the workgroup loads its assigned portion
of the tile as defined by the distribution.

### Pipeline: descriptor -> view -> distribution -> window -> **load**

## Unified API

```cpp
// Load tile, returns a distributed tensor (per-thread register buffer)
auto tile = ops::load_tile(window);

// Load into existing tensor
ops::load_tile(existing_tile, window);
```

## Backend Mapping

| Unified | CK_TILE | MINT |
|---------|---------|------|
| `load_tile(window)` | `ck_tile::load_tile(window)` | `masked_load_no_shuffle_vectorized<vec_dims, vec_lens, freeze>(win, mask)` |

## Key Design Notes

- For MINT, the vector dims, vector lengths, and freeze dims are **auto-derived**
  from the window's distribution:
  - Vector dim = last element dim alias (inner/contiguous)
  - Vector length = element length of that dim (= VecSize from config)
  - Freeze dims = cross-freeze: when iterating dim0, freeze dim1 and vice versa
- This matches the pattern used in the GEMM XDL example
- No mask is applied (empty `tuple<>{}`)

## Verification

Source filled with row pattern: `a[row][col] = row + 1`.
After loading, we verify:
1. All loaded values are in valid range [1, M]
2. No zeros (would indicate OOB or failed load)
3. Global sum across all threads matches expected `K * sum(1..M)`
4. All per-thread sums are positive

## Build & Run

```bash
ninja example_unified_tile_04_load_ck
./bin/example_unified_tile_04_load_ck

ninja example_unified_tile_04_load_mint
./bin/example_unified_tile_04_load_mint
```
