# CK Tile Distribution Encoding Tutorial

## Overview

Every `load_tile` and `store_tile` in CK needs to know **which thread reads which data element**.
This mapping is defined by a `tile_distribution_encoding` — a compile-time struct with 6 template
parameters:

```cpp
tile_distribution_encoding<Rs, Hs, Ps_major, Ps_minor, Ys_major, Ys_minor>
```

Every level of **Hs** (hierarchical dimensions) is assigned to exactly one role:

| Role | Meaning |
|------|---------|
| **P** (parallel) | Thread ID selects which slice — different threads get different data |
| **Y** (yield) | Each thread owns the entire range in its buffer |
| **R** (replicate) | Identical data broadcast to multiple thread groups |

This tutorial series makes these mappings concrete: each scenario loads a matrix using a
production distribution and prints what each thread received, so you can trace the encoding
by hand.

No compute is performed. The tutorials are purely about data movement and thread-to-data mapping.

**Architecture note:** All comments, ASCII art, and concrete values in these tutorials assume
**CDNA (warp_size=64)**. On RDNA (warp_size=32), the thread-to-data mapping will differ:
`lane_id` ranges 0–31 instead of 0–63, `BlockSize / warp_size` yields more warps, and
production code derives different values for M0/M1/M2/K0/K1. The encodings will compile and
run on both architectures, but the printed output will not match the comments on RDNA.

## Tutorials

The recommended reading order is **1 → 3 → 2 → 4** (simple to complex):

| File | Scenario | Tile | Threads | Key Concept | Source |
|------|----------|------|---------|-------------|--------|
| `tile_distribution.cpp` | 1 | 32×8 | 128 (2 warps) | NDimP=1, basic P/Y split | MakeADramTileDistribution (simplified) |
| `tile_distribution_3.cpp` | 3 | 128×8 | 128 (2 warps) | NDimP=2, lane→M only (small K) | MakeADramTileDistribution (RowMajor) |
| `tile_distribution_2.cpp` | 2 | 64×32 | 128 (2 warps) | NDimP=2, lane→M+K (coalesced) | MakeADramTileDistribution (RowMajor) |
| `tile_distribution_4.cpp` | 4 | 128×8 | 256 (4 warps) | NDimP=2 + R (inter-warp replicate) | Tutorial replicate-pattern example |

Scenarios 1-3 are from
`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`.
Scenario 4 demonstrates the R (replicate) dimension pattern used for scale-factor style
distributions, but this tutorial does not currently point to a matching production helper in
this repository.

## Building

```bash
cd <repo-root>/projects/composablekernel/build

# Build all tutorials:
ninja tutorials

# Or build individually:
ninja tile_tutorial_tile_distribution
ninja tile_tutorial_tile_distribution_2
ninja tile_tutorial_tile_distribution_3
ninja tile_tutorial_tile_distribution_4
```

## Reference

The distribution encoding is defined in:
- `include/ck_tile/core/tensor/tile_distribution_encoding.hpp`
- `include/ck_tile/core/tensor/tile_distribution.hpp` (line 98: NDimP → thread identity)
- `include/ck_tile/core/algorithm/coordinate_transform.hpp` (merge_v3_division_mod: the modular decomposition)

Production usage:
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`
  (`MakeADramTileDistribution`, `MakeBDramTileDistribution`)
