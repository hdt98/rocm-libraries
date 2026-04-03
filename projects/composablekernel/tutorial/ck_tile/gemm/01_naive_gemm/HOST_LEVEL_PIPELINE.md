# Host-Level Pipeline: GridGemm

This document explains `GridGemm` (in `host_level/grid_gemm.hpp`), which orchestrates the
distribution of work across thread blocks and manages the high-level flow of the GEMM computation.

## Overview

`GridGemm` runs on the GPU (one instance per thread block). It is responsible for:
1. **Extracting problem dimensions** from tensor descriptors at runtime
2. **Block-to-tile mapping**: Converting a linear block ID to a 2D tile coordinate `(iM, iN)`
3. **Creating tile windows**: Establishing initial windows over A and B in DRAM
4. **Delegating computation**: Calling the block-level pipeline
5. **Epilogue + store**: Casting accumulator, applying `CElementFunction`, writing C to DRAM

```cpp
template <typename Problem, typename Policy>
struct GridGemm
{
    template <typename AGridTensorView, typename BGridTensorView, typename CGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   CGridTensorView& c_grid,
                                   const CElementFunction& c_element_func) const
    { ... }
};
```

---

## Step 1: Extract Problem Dimensions

```cpp
// Extract M, N, K from the tensor descriptors at runtime.
// These are runtime values (M, N, K were passed as kernel arguments and baked into the descriptors).
const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});  // rows of A
const auto N = c_grid.get_tensor_descriptor().get_length(number<1>{});  // cols of C
const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});  // cols of A
```

The tensor descriptors created in `Gemm::operator()` (the kernel entry point) carry the
runtime values M, N, K. Extracting them here via `get_length()` avoids passing them as
additional function arguments through the hierarchy.

---

## Step 2: Map Thread Block to Tile Coordinates

```cpp
// get_block_id(): AMD GPU intrinsic equivalent to blockIdx.x in CUDA.
// The grid is 1D (kGridSize blocks total), so block IDs run from 0 to kGridSize-1.
const auto id_block = get_block_id();

// Number of tiles needed to cover M and N.
// integer_divide_ceil(x, y) = (x + y - 1) / y — handles non-divisible sizes.
const auto num_tile_m = integer_divide_ceil(M, kMPerBlock);
const auto num_tile_n = integer_divide_ceil(N, kNPerBlock);

// MakeBlock2TileMap returns a lambda: block_id → (m_tile_idx, n_tile_idx).
const auto block2tile = Policy::template MakeBlock2TileMap<Problem>(num_tile_m, num_tile_n);
const auto id_tile    = block2tile(id_block);
```

### N-First Tile Ordering

`MakeBlock2TileMap` uses N as the fast (inner) dimension:
```
block_id 0 → (m=0, n=0)
block_id 1 → (m=0, n=1)
block_id 2 → (m=0, n=2)
...
block_id num_tile_n     → (m=1, n=0)
block_id num_tile_n + 1 → (m=1, n=1)
...
```

Consecutive block IDs map to adjacent N tiles **within the same M row**. This is N-first
(row-major in tile space) ordering.

**Why N-first?** Adjacent blocks in the same M row all read the **same A rows** but
different B rows. Since A data (same M strip) stays in L2 cache while N tiles are processed,
N-first ordering maximizes L2 reuse for A. B is read once per block and not reused.

**Example for default problem (M=3328, N=4096, kMPerBlock=256, kNPerBlock=128):**
- `num_tile_m = ceil(3328/256) = 13`
- `num_tile_n = ceil(4096/128) = 32`
- Blocks 0–31 all cover M rows 0–255 (same A strip), N cols 0–4095 (all of N)
- Block 32 begins the next M strip (rows 256–511)

---

## Step 3: Promote Tile Origin to SGPRs

```cpp
// __builtin_amdgcn_readfirstlane: broadcasts a VGPR value from lane 0 to all lanes,
// making the compiler treat iM/iN as SGPRs rather than VGPRs.
//
// Why? The tile origin (iM, iN) is the same for every thread in the block — it's a
// uniform, block-level value. Keeping it in VGPRs wastes 64 VGPR slots (one per lane).
// Promoting to SGPR reduces VGPR pressure significantly and allows scalar address
// arithmetic for window origins.
const auto iM = __builtin_amdgcn_readfirstlane(id_tile.template at<0>() * kMPerBlock);
const auto iN = __builtin_amdgcn_readfirstlane(id_tile.template at<1>() * kNPerBlock);
```

This is an AMD GCN/CDNA-specific optimization: `readfirstlane` reads the value from lane 0
and broadcasts it to all 64 lanes as a scalar register. Since `id_tile` is the same for all
threads in a block (derived only from `get_block_id()`), this is always safe.

---

## Step 4: Create Block Windows over A and B

```cpp
// A block window: covers rows [iM, iM+kMPerBlock) and K cols [0, kKPerBlock) initially.
// The block pipeline will slide this window along K in each loop iteration.
auto a_block_window = make_tile_window(
    a_grid,
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {iM, 0});   // origin: row iM, K column 0

// B block window: covers rows [iN, iN+kNPerBlock) and K cols [0, kKPerBlock).
// B is stored as [N, K] (N as leading dimension, K contiguous), so this window
// selects the N-strip [iN, iN+kNPerBlock) and the first K slice.
auto b_block_window = make_tile_window(
    b_grid,
    make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
    {iN, 0});   // origin: N-row iN, K column 0
```

Both windows start at K column 0. The block pipeline's inner loop advances the K origin
by `kKPerBlock` each iteration using `move_tile_window`.

**Tile windows for the default problem (block 0, tile (0,0)):**
```
Matrix A (3328 × 4096):            Matrix B (4096 × 4096, stored [N,K]):
┌─────────────────────────────┐    ┌──────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━┓            │    │ ┏━━━━━━━━━━━━━┓              │
│ ┃ a_block_window┃            │    │ ┃b_block_window┃             │
│ ┃ 256×32        ┃ → slides → │    │ ┃ 128×32       ┃ → slides →  │
│ ┃ K=0:31        ┃            │    │ ┃ K=0:31       ┃             │
│ ┗━━━━━━━━━━━━━━━┛            │    │ ┗━━━━━━━━━━━━━┛              │
│   (rows 0–255)               │    │  (N-rows 0–127)              │
└─────────────────────────────┘    └──────────────────────────────┘
```

---

## Step 5: Allocate LDS and Call Block Pipeline

```cpp
// GetStaticLdsSize(): compile-time constant — the compiler allocates exactly this many
// bytes as __shared__ memory for the block. No runtime allocation, no memory management.
// The LDS size = A_tile (16KB, rounded to 16B) + B_tile (8KB) = 24KB for default params.
constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();
__shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

// Run the block pipeline: iterates K/kKPerBlock times, loading A and B from DRAM
// into LDS each iteration, computing MFMA, and accumulating into acc_block_tile.
// acc_block_tile is in AccDataType (float) precision, held in VGPRs.
const auto acc_block_tile =
    block_gemm_pipeline(a_block_window, b_block_window, K / kKPerBlock, p_smem_char);
```

`block_gemm_pipeline` is a compile-time constant — it's just a type tag, not a runtime
object. The actual pipeline is `BlockGemmPipelineAGmemBGmemCReg`, the non-prefetch pipeline.

`K / kKPerBlock` assumes K is divisible by `kKPerBlock` (a precondition of this tutorial kernel).
For production kernels handling arbitrary K, `integer_divide_ceil` would be used with boundary
handling.

---

## Step 6: Epilogue + Store C to DRAM

```cpp
// Cast accumulator tile from AccDataType (float) to CDataType (half_t), apply epilogue.
// tile_elementwise_in applies the lambda element-wise over the distributed tile,
// producing a new tile of the output type.
// type_convert<CDataType>(acc): FP32 → FP16 conversion (round-to-nearest-even).
// c_element_func(x): apply CElementFunction (identity in this tutorial).
const auto c_block_tile = tile_elementwise_in(
    [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
    acc_block_tile);

// Create the C tile window at this block's output position.
// store_tile uses the C tile's distribution to scatter per-thread register data
// back to the correct global memory addresses using vectorized stores.
auto c_window = make_tile_window(
    c_grid,
    make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
    {iM, iN});  // origin: block's assigned output tile

store_tile(c_window, c_block_tile);
```

### What `store_tile` does

`store_tile` is the inverse of `load_tile`: it reads each thread's register slice from
`c_block_tile`, computes the corresponding DRAM address using the C tile's distribution,
and issues vectorized stores (`global_store_dwordx4`). Adjacent threads in a warp write to
adjacent memory locations — coalesced stores with zero wasted bandwidth.

The C tile window has **no explicit distribution** when created here; `store_tile` extracts
it from `c_block_tile`'s tile distribution internally.

---

## Complete Flow for Block 0 (default sizes)

```
1. id_block = 0
2. num_tile_m = 13, num_tile_n = 32
3. id_tile = block2tile(0) = (m=0, n=0)
4. iM = 0, iN = 0  (promoted to SGPRs)
5. a_block_window: rows 0–255, K=0–31 initially
   b_block_window: N-rows 0–127, K=0–31 initially
6. Allocate __shared__ char p_smem_char[24576]
7. Call block_gemm_pipeline(a_window, b_window, 128, p_smem_char)
   → 128 K-iterations (K=4096, kKPerBlock=32 → 128 slices)
   → Each iteration: DRAM→VGPRs→LDS→sync→MFMA→sync
   → Returns acc_block_tile (256×128 in FP32 VGPRs)
8. tile_elementwise_in: FP32→FP16, apply CElementFunction (identity)
9. c_window at (0, 0), size 256×128
10. store_tile: write c_block_tile from VGPRs to C[0:255, 0:127]
```

All 416 blocks execute steps 1–10 in parallel, each computing its assigned 256×128 tile.

---

## Block-to-Tile Mapping Example

For `num_tile_m=13, num_tile_n=32` (416 blocks total, N-first ordering):

```
Block ID → (M tile, N tile) → C region written
       0 → (0,  0) → C[0:255,     0:127]
       1 → (0,  1) → C[0:255,   128:255]
       2 → (0,  2) → C[0:255,   256:383]
      ...
      31 → (0, 31) → C[0:255,  3968:4095]
      32 → (1,  0) → C[256:511,   0:127]
      33 → (1,  1) → C[256:511, 128:255]
      ...
     415 → (12, 31) → C[3072:3327, 3968:4095]
```

N-first: blocks 0–31 all share the same A rows (rows 0–255), so A data stays in L2 cache
while the kernel sweeps through all 32 N tiles.

---

## Key Concepts Summary

### 1. Grid-Level Tile Coverage

`GridGemm` partitions the M×N output matrix C into `num_tile_m × num_tile_n` tiles.
Each of the `kGridSize` blocks computes exactly one tile of C.

### 2. Block-to-Tile Mapping (N-first)

`MakeBlock2TileMap` produces N-first (row-major in tile space) ordering. This is an L2
cache optimization: all blocks in the same M-strip share the same A rows.

### 3. SGPR Promotion via `readfirstlane`

Converting uniform (block-wide) values from VGPRs to SGPRs saves 64 VGPR slots per value
and enables scalar address arithmetic for window origins.

### 4. Tile Windows

Windows carry: underlying tensor view + window size (compile-time) + origin (runtime). They
slide along K inside the block pipeline via `move_tile_window`. The C window is write-only
and created only after the block pipeline completes.

### 5. Epilogue Chain

The accumulator (FP32) is converted to C's element type (FP16) and passed through
`CElementFunction` in a single `tile_elementwise_in` call before storing to DRAM. This
separation of accumulation from output formatting is what makes the epilogue extensible:
any per-element transformation (bias add, activation, scaling) can be plugged in as
`CElementFunction` without modifying the GEMM core.

### 6. LDS Size is Compile-Time

`GetStaticLdsSize()` is `constexpr` — the `__shared__` array size is a compile-time constant.
The compiler can verify it fits within the 64KB LDS limit and allocate it statically.
