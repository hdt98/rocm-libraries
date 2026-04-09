# Block-Level Pipeline: BlockGemmPipelineAGmemBGmemCReg

## Overview

The **Block-Level Pipeline** (`BlockGemmPipelineAGmemBGmemCReg`) is where the actual GEMM
computation happens for one block tile. It orchestrates:
1. **Data movement** from DRAM → per-thread VGPRs → LDS
2. **GEMM computation** using MFMA instructions on data in LDS
3. **Iteration** over the K dimension in slices of `kKPerBlock`

This pipeline is called by `GridGemm` after it has mapped the current block to its `(iM, iN)`
tile and created `a_block_window` and `b_block_window` over global memory
(passed in as `a_dram_block_window_tmp` and `b_dram_block_window_tmp`).

---

## Architecture: Problem and Policy

Like all components in CK Tile, the block pipeline follows the **Problem/Policy** pattern:

### Problem: `BlockGemmPipelineProblem`
Contains:
- **Data types**: `ADataType`, `BDataType`, `CDataType`
- **Block size**: `kBlockSize` — used to derive DRAM tile distributions
- **Block GEMM shape**: `BlockGemmShape` with `kM`, `kN`, `kK`

Note: `AccDataType` and `CElementFunction` are not in `BlockGemmPipelineProblem` — they are
applied at the grid level (`GridGemm`) after the pipeline returns `c_block_tile`.

### Policy: `BlockGemmPipelineAGmemBGmemCRegPolicy`
Contains strategies for:
1. **DRAM tile distributions** (`MakeADramTileDistribution`, `MakeBDramTileDistribution`)
   — Defines how 256 threads map to elements of a block tile for parallel vectorized loads
2. **LDS layout descriptors** (`MakeALdsBlockDescriptor`, `MakeBLdsBlockDescriptor`)
   — Describes how data is laid out in LDS for `ds_write_b128` / MFMA compatibility
3. **Block GEMM** (`GetBlockGemm`) — Returns `BlockGemmASmemBSmemCReg`, the warp-level MFMA coordinator

---

## Inputs and Outputs

```cpp
template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
CK_TILE_HOST_DEVICE auto operator()(
    const ADramBlockWindowTmp& a_dram_block_window_tmp,   // window over A in DRAM (MPerBlock × KPerBlock)
    const BDramBlockWindowTmp& b_dram_block_window_tmp,   // window over B in DRAM (NPerBlock × KPerBlock)
    index_t num_loop,                                      // K / kKPerBlock iterations
    void* p_smem) const                                   // pointer to block's shared memory (LDS)
```

The `[[maybe_unused]]` attribute on the window parameters in `BlockGemmASmemBSmemCReg::operator()`
(the warp-level layer below) suppresses compiler warnings when the parameters are only used for
their type information at compile time — the warp layer builds its own windows from the tensor view.

### Output:
- Returns `c_block_tile`: a `static_distributed_tensor` containing the accumulated C tile
  in `AccDataType` (float) distributed across per-thread VGPRs.

---

## Step-by-Step Walkthrough

### Step 1: Create LDS Tensor Views

```cpp
// A region in shared memory
ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

// a_lds_block_space_size_aligned: A's LDS region rounded up to 16 bytes.
// B is placed immediately after, so B's pointer is 16-byte aligned —
// satisfying ds_write_b128 alignment requirements.
constexpr index_t a_lds_block_space_size_aligned =
    integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) * 16;

// B region: placed after A's aligned region
BDataType* p_b_lds = static_cast<BDataType*>(
    static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
```

**Memory Layout in LDS:**
```
Shared Memory (LDS):
┌──────────────────────────────┬────────────────────┐
│   A Block Tile               │   B Block Tile      │
│   (256×32 fp16 = 16 KB)      │   (128×32 fp16 = 8 KB)│
│   16-byte aligned            │                    │
└──────────────────────────────┴────────────────────┘
↑ p_a_lds                      ↑ p_b_lds (16-byte aligned)
```

`GetStaticLdsSize()` (called by `GridGemm` before allocating `__shared__`) computes this
at compile time:
```cpp
sizeof(A_lds) rounded up to 16 + sizeof(B_lds)
= integer_divide_ceil(2 * 256 * 32, 16) * 16 + 2 * 128 * 32
= 16384 + 8192 = 24576 bytes
```

---

### Step 2: Create Tile Windows for Data Movement

We create **6 tile windows** serving different purposes:

#### 2a. A DRAM → Registers (load distribution)

```cpp
auto a_copy_dram_window = make_tile_window(
    a_dram_block_window_tmp.get_bottom_tensor_view(),
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    a_dram_block_window_tmp.get_window_origin(),
    Policy::template MakeADramTileDistribution<Problem>());  // ← tile distribution
```

The **tile distribution** tells each of the 256 threads which elements of the 256×32 A tile
to load. Thread T does not load a contiguous row; instead it loads M0=4 groups of K1=8
elements (see `TILE_DISTRIBUTION.md` for the derivation). Each group is one 128-bit vector
load, so Thread T issues 4 × 128-bit `global_load_dwordx4` instructions total.

#### 2b. A Registers → LDS (store window; same distribution)

```cpp
auto a_copy_lds_window = make_tile_window(
    a_lds_block,
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {0, 0},
    a_copy_dram_window.get_tile_distribution());  // ← same distribution as DRAM
```

Using the **same distribution** for the LDS store as the DRAM load is critical: it guarantees
that Thread T stores the elements it just loaded from DRAM to the same logical positions in LDS.
A mismatch between load and store distributions would silently corrupt the LDS layout and
produce wrong MFMA results.

#### 2c. A LDS → MFMA (no distribution)

```cpp
auto a_lds_gemm_window = make_tile_window(
    a_lds_block,
    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
    {0, 0});  // no tile distribution
```

This window has **no explicit tile distribution**. The warp-level GEMM (`BlockGemmASmemBSmemCReg`)
applies its own warp-level distribution internally when creating warp sub-windows. The GEMM
layer does not need the DRAM distribution here.

**Similarly for B:**
- `b_copy_dram_window`: Load B from DRAM (with distribution)
- `b_copy_lds_window`: Store B to LDS (same distribution as DRAM)
- `b_lds_gemm_window`: Read B from LDS for GEMM (no distribution)

---

### Step 3: Create Distributed Tensors (per-thread VGPRs)

```cpp
// Types derived from distributions at compile time
using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

ABlockTile a_block_tile;  // Per-thread register buffer for A (lives in VGPRs)
BBlockTile b_block_tile;  // Per-thread register buffer for B (lives in VGPRs)
```

`make_static_distributed_tensor` creates a **compile-time distributed register buffer**:
- Each thread holds a different slice of the tile in its own VGPRs
- Buffer size = M0 × K1 = 4 × 8 = 32 fp16 elements per thread for A
- No two threads share the same element

Collectively, all 256 threads hold the entire 256×32 tile (8192 elements), each thread owning
its 32-element slice.

```cpp
// c_block_tile: type inferred from the init form of BlockGemm's operator().
// This avoids having to spell out the C distribution encoding explicitly here.
auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};
```

The C block tile type is derived using `decltype` applied to the block GEMM's return type —
a useful trick that lets the compiler deduce the complex C distribution type automatically.

---

### Step 4: The GEMM Loop (Non-Prefetch Pipeline)

```cpp
// Initialize C accumulator to zero before the K loop
tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

index_t iCounter = num_loop;  // num_loop = K / kKPerBlock

while(iCounter > 0)
{
    // Step 1: Load A and B tiles from DRAM into per-thread registers (VGPRs).
    // Each thread loads its assigned elements using vectorized global loads.
    a_block_tile = load_tile(a_copy_dram_window);
    b_block_tile = load_tile(b_copy_dram_window);

    // Step 2: Advance DRAM windows to the next K slice.
    // Done before the LDS store so the compiler can schedule address computation
    // while the stores are in flight.
    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);  // {0, kKPerBlock}
    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);  // {0, kKPerBlock}

    // Step 3: Store A and B tiles from VGPRs to LDS.
    // Thread T stores exactly the elements it loaded — same distribution → correct layout.
    store_tile(a_copy_lds_window, a_block_tile);
    store_tile(b_copy_lds_window, b_block_tile);

    // Step 4: Barrier 1 — wait for ALL threads to finish writing to LDS.
    // Without this, a thread that finishes early could start reading LDS for MFMA
    // while another thread is still writing — producing stale data.
    block_sync_lds();

    // Step 5: Block GEMM: c_block_tile += A_lds × B_lds using MFMA instructions.
    // The warp-level GEMM reads A and B from LDS via warp sub-windows and
    // accumulates into c_block_tile (in FP32 VGPRs).
    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

    // Step 6: Barrier 2 — wait for ALL threads to finish reading LDS (MFMA complete).
    // Without this, the next iteration's LDS store could overwrite data that another
    // thread is still consuming for MFMA.
    block_sync_lds();

    iCounter--;
}

return c_block_tile;  // FP32 VGPRs returned to GridGemm for cast + epilogue + store
```

This is the **non-prefetch pipeline**: each iteration fully serializes load → store → sync →
MFMA → sync. No double-buffering, no overlap between data movement and computation. This is
the simplest correct pipeline — hence "naive".

---

## Memory Flow Diagram

```
Iteration i (K slice i):

DRAM A[iM:iM+256, i*32:(i+1)*32]
         │
         │ load_tile (parallel, vectorized)
         ↓
  VGPRs: a_block_tile (per-thread, M0×K1=32 fp16 elements each)
         │
         │ store_tile
         ↓
  LDS A region (256×32 fp16, shared by all 256 threads)
         │
         │ block_sync_lds() ← Barrier 1
         │
         ↓
  BlockGemmASmemBSmemCReg
    reads A from LDS via a_lds_gemm_window (warp sub-tiles)
    reads B from LDS via b_lds_gemm_window (warp sub-tiles)
    c_block_tile += A_warp × B_warp  (MFMA instructions, FP32 accumulation)
         │
         │ block_sync_lds() ← Barrier 2
         │
         ↓
  c_block_tile in VGPRs (FP32, accumulates across all K iterations)
```

---

## Thread Ownership Example (for A tile load)

With the DRAM tile distribution (M0=4, M1=4, M2=16, K0=4, K1=8):

Thread T (warp W, lane L within warp):
- `M2` position within warp: `L / K0 = L / 4`
- `K0` position within warp: `L % K0 = L % 4`
- `M1` position: warp index W
- Each thread loads M0=4 separate K1=8 element chunks (one per M0 repetition)

Concretely, Thread 0 (warp 0, lane 0):
- M2=0, K0=0 → loads rows {0, 64, 128, 192} (one per M0 iteration), columns 0–7 each
- Issues 4 × `global_load_dwordx4` instructions, loading A[0,0:7], A[64,0:7], A[128,0:7], A[192,0:7]

This is not a single row — each thread loads a strided pattern covering multiple M rows
and a fixed K group, designed so all 64 threads in the warp together cover the same set of
K columns with perfectly coalesced (contiguous, cache-line aligned) accesses.

---

## Key Concepts Summary

### 1. Tile Distribution
- Maps each of the 256 threads to specific elements in the 256×32 tile
- Enables **parallel, vectorized** DRAM loads (128-bit per call, 4 calls per thread)
- The **same distribution** governs both the DRAM load and the LDS store

### 2. Static Distributed Tensor (per-thread VGPRs)
- `a_block_tile`, `b_block_tile`: staging area between DRAM and LDS
- Each thread owns a different, non-overlapping slice of the tile
- All indexing resolved at compile time — zero runtime overhead

### 3. Tile Window Movement
- Windows slide along the K dimension via `move_tile_window(window, {0, kKPerBlock})`
- The `a_copy_lds_window` and `b_copy_lds_window` do NOT move — LDS is reused each iteration

### 4. LDS as Staging Area
- Shared memory accessible to all 256 threads in the block
- Required because MFMA needs all threads in a warp to access sub-tiles of A and B
- Reused across K iterations (same physical LDS buffer, overwritten each iteration)

### 5. Two-Barrier Pattern
- **Barrier 1** (`block_sync_lds()` after store): Ensures all LDS writes complete before MFMA reads
- **Barrier 2** (`block_sync_lds()` after MFMA): Ensures all MFMA reads complete before next iteration overwrites LDS
- Both barriers are necessary for correctness; removing either causes data races

### 6. Non-Prefetch vs. Prefetch Pipelines
This is the **non-prefetch** (naive) pipeline: the DRAM load for iteration `i+1` only begins
after iteration `i`'s barriers and MFMA complete. A prefetch pipeline would overlap the
iteration `i+1` load with iteration `i`'s MFMA using double-buffering in LDS. That
optimization is a natural next step from this baseline.
