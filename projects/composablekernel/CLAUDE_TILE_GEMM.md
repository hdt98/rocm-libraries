# CLAUDE.md - Tile-Based GEMM Programming Guide

This file provides guidance to Claude Code when working with CK_Tile and MINT GEMM implementations in this repository.

---

## Project Overview

This project contains **two tile-based GEMM kernel programming frameworks**:

1. **CK_Tile** - AMD's hierarchical tile programming model optimized for ROCm/HIP
2. **MINT (Metal INterface Tiles)** - Portable polymorpher-based tile programming model for ROCm and CUDA

Both implement high-performance matrix multiplication (GEMM) using different architectural philosophies.

---

## Directory Structure

```
composablekernel/
├── include/
│   ├── ck_tile/ops/gemm/          # CK_Tile GEMM implementation
│   │   ├── kernel/                # Kernel entry points
│   │   ├── pipeline/              # Pipeline implementations
│   │   ├── block/                 # Block-level GEMM
│   │   ├── warp/                  # Warp-level GEMM
│   │   └── ...
│   └── mint/                      # MINT framework
│       ├── core/                  # Core utilities
│       ├── poly/                  # Polymorpher algebra
│       ├── tensor/                # Tensor abstractions
│       └── tile/                  # Tile operations
│
├── example/
│   └── ck_tile/
│       ├── 03_gemm/               # CK_Tile GEMM examples
│       │   └── universal_gemm.cpp
│       └── mint_example/          # MINT examples
│           └── gemm_xdl_v1/
│               └── rocm_gemm_xdl_v1.cpp
│
└── TILE_WINDOW_ARCHITECTURE.md   # Architecture comparison document
```

---

## Key Files and Their Purposes

### CK_Tile GEMM Implementation

The CK_Tile GEMM follows a **hierarchical architecture** with clear separation between layers:

#### 1. Kernel Layer (Entry Point)
**File:** [`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp)

**Purpose:**
- Defines the universal GEMM kernel template
- Handles workgroup positioning and block-level tiling
- Creates block windows (logical tile views without thread distribution)
- Delegates to pipeline for actual computation

**Key Functions:**
```cpp
// Creates block-level windows at lines ~820-894
MakeABlockWindows() // Creates A tensor block window
MakeBBlockWindows() // Creates B tensor block window
MakeDBlockWindows() // Creates D tensor block window (bias/residual)
MakeCBlockWindows() // Creates C tensor block window (output)

// Main kernel operator at lines ~1220-1256 (non-persistent)
// or lines ~1258-1340 (persistent variant)
operator()(KernelArgs kargs)
```

**What to look for:**
- Block window creation (lines 820-1109)
- Workgroup coordinate calculation
- Block-level logical tiling
- No thread-level distribution at this layer

#### 2. Pipeline Layer (Memory Access Patterns)
**File:** [`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp)

**Purpose:**
- Implements the GEMM computation pipeline
- Defines data movement: GMEM → VGPR → LDS → MFMA
- Creates thread-distributed copy windows from block windows
- Handles prefetching, loop scheduling, and tail handling

**Key Components:**
```cpp
// Pipeline metadata (lines 18-30)
PrefetchStages = 2    // Global prefetch stages
PrefillStages = 1     // LDS prefill stages
GlobalBufferNum = 1   // Number of LDS buffers

// Main pipeline operator (lines 401-626)
operator()(a_dram_block_window,      // From kernel layer
           a_element_func,
           b_dram_block_window,
           b_element_func,
           num_loop,
           p_smem)

// Hot loop scheduler (lines 252-390)
HotLoopScheduler() // Instruction scheduling for MFMA overlap
```

**What to look for:**
- Copy window creation from block windows (lines 452-459)
- Global memory loads (lines 480-494, 522-526)
- LDS writes (lines 496-518)
- MFMA computation (line 573)
- Sync points (lines 528, 575)

#### 3. Pipeline Base (Window Creation Logic)
**File:** [`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp)

**Purpose:**
- Provides base functionality for pipeline implementations
- Creates thread-distributed copy windows from block windows
- Implements window transformation helpers

**Key Functions:**
```cpp
// Lines 152-291: Create A tensor windows
GetAWindows(a_dram_block_window_tmp,  // Block window from kernel
            a_lds_block_view,
            a_lds_load_tile_distr)
    Returns: (a_copy_dram_window,  // For GMEM→VGPR
              a_copy_lds_window,    // For VGPR→LDS
              a_lds_gemm_window)    // For LDS→MFMA

// Lines 155-173: Copy DRAM window
CopyADramWindow() // Adds thread distribution to block window

// Lines 238-268: Make LDS windows
MakeALdsWindows() // Creates LDS store and load windows
```

**Important:** This is where the **second window** is created by adding thread distribution to the block window!

#### 4. Policy Layer (Thread Distribution)
**File:** [`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp)

**Purpose:**
- Defines HOW to access memory (policies)
- Creates thread distributions for optimal memory access
- Configures LDS descriptors with bank conflict avoidance

**Key Functions:**
```cpp
// Lines 124-312: A tensor LDS descriptor
MakeALdsBlockDescriptor<Problem>()

// Lines 321-510: B tensor LDS descriptor
MakeBLdsBlockDescriptor<Problem>()

// Lines 714-748: A tensor DRAM distribution
MakeADramTileDistribution<Problem>()
    Uses: tile_distribution_encoding_pattern_2d
    Patterns: thread_raked, thread_striped

// Lines 752-791: B tensor DRAM distribution
MakeBDramTileDistribution<Problem>()

// Lines 570-601: Vector size calculation
GetVectorSizeA<Problem>()
GetVectorSizeB<Problem>()
```

**What to look for:**
- LDS layout optimization (XOR transforms for bank conflict avoidance)
- Thread distribution patterns (raked vs striped)
- Automatic vectorization size calculation
- Hardware-aware optimizations (wave size, LDS banks)

#### 5. Block-Level GEMM
**File:** [`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp)

**Purpose:**
- Block-level GEMM interface
- Coordinates warp-level GEMM operations
- Manages accumulator distribution

**Key Components:**
- Warp gemm instantiation
- Block-level accumulator management
- Prefetch logic for LDS reads

---

### MINT GEMM Implementation

MINT implements everything in a **single file** with explicit polymorpher composition.

#### Single-File Implementation
**File:** [`example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp`](~/repos/rocm-libraries/projects/composablekernel/example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp)

**Key Sections:**

1. **Block Copy Distribution (Lines 164-268)**
```cpp
make_block_matmul_a_m_k_distribution<BlockSize, MPerBlock, KPerBlock, KPack>()
    - Explicit thread partitioning: tid_m × tid_k
    - M dimension split: M0 × M1
    - K dimension split: K0 × K1
    - Polymorpher composition: merge + split transformations
    - Returns: distributed_tensor_descriptor
```

2. **Warp Matmul Distribution (Lines 277-463)**
```cpp
make_warp_matmul_a_m_k_distribution<MPerWarp, KPerWarp, KPack, MRepeat, KRepeat>()
    - Warp-level thread distribution
    - Repeat factors for larger accumulation
    - Element-level access pattern
```

3. **LDS Tensor Descriptors (Lines 466-581)**
```cpp
make_a_smem_tensor_descriptor<MPerBlock, KPerBlock, KPack>()
    - XOR rotation for bank conflict avoidance
    - Packed layout: M0 × M1 × K0 × K1
    - Transform to logical M × K layout
```

4. **Main Kernel (Lines 600-1000)**
```cpp
gpu_gemm_kernel<T, TAcc, ThreadPerBlock, MPerBlock, NPerBlock, KPerBlock, ...>()
    - Line 614-623: Create GMEM descriptors
    - Line 626-637: Create distributions
    - Line 639-696: Create element layouts
    - Line 756-769: Create distributed windows (GMEM)
    - Line 772-785: Create distributed windows (LDS)
    - Line 796-809: Create warp windows
    - Line 853-880: Block GMEM load + LDS store (stage 0)
    - Line 885-934: Main loop (GMEM load, LDS store, MFMA)
    - Line 938-954: Tail
    - Line 966-999: Store results to GMEM
```

**What to look for:**
- Single distributed window creation (position + distribution in one call)
- Explicit polymorpher algebra
- No hierarchical layers - direct mapping
- M-strided thread distribution pattern

---

## CK_Tile Code Flow

### Execution Path

```
1. Kernel Entry Point
   └─> universal_gemm_kernel.hpp::operator()
       │
       ├─> Calculate workgroup position (i_m, i_n)
       │
       ├─> MakeABlockWindows()  ──┐
       ├─> MakeBBlockWindows()    │ Block windows (no thread distribution)
       └─> MakeDBlockWindows()  ──┘
           │
           │ Pass block windows to pipeline
           ▼
2. Pipeline Layer
   └─> gemm_pipeline_ag_bg_cr_comp_v3.hpp::operator()
       │
       ├─> Base::GetAWindows(a_block_window, ...)
       │   └─> CopyADramWindow()  ──┐
       │       └─> Policy::MakeADramTileDistribution()  ← ADD thread distribution
       │
       │   └─> MakeALdsWindows()
       │       Returns: (dram_copy_window, lds_copy_window, lds_gemm_window)
       │
       ├─> load_tile_with_elementwise(a_dram_copy_window)  ← GMEM → VGPR
       ├─> store_tile(a_lds_copy_window)                   ← VGPR → LDS
       ├─> block_sync_lds()
       ├─> BlockGemm.LocalPrefetch(a_lds_gemm_window)      ← LDS → VGPR
       │
       └─> Main Loop:
           ├─> load_tile (GMEM → VGPR)
           ├─> block_gemm (MFMA)
           ├─> block_sync_lds()
           ├─> store_tile (VGPR → LDS)
           └─> HotLoopScheduler() ← Instruction scheduling
```

### Two-Window Creation

```
Block Window (Kernel Layer)
  ├─> make_tensor_view(global_memory, descriptor)
  ├─> pad_tensor_view(view, tile_size, pad_flags)
  └─> make_tile_window(pad_view, tile_size, {i_m, 0})
      │
      │ No thread distribution - pure logical positioning
      │
      ▼
Copy Windows (Pipeline Layer)
  ├─> GetAWindows(block_window, lds_view, load_distribution)
  │   │
  │   ├─> CopyADramWindow(block_window)
  │   │   └─> make_tile_window(
  │   │         block_window.get_bottom_tensor_view(),
  │   │         tile_size,
  │   │         block_window.get_window_origin(),
  │   │         Policy::MakeADramTileDistribution())  ← THREAD DISTRIBUTION ADDED
  │   │
  │   ├─> MakeALdsWindows(lds_view)
  │   │   ├─> LDS store window (coalesced writes)
  │   │   └─> LDS load window (MFMA-friendly reads)
  │   │
  │   └─> Returns 3 windows from 1 block window
```

---

## MINT Code Flow

### Execution Path

```
1. Kernel Entry Point
   └─> gpu_gemm_kernel<...>()
       │
       ├─> Create GMEM descriptors
       │   └─> make_aliased_naive_packed_tensor_descriptor(aliases<"M", "K">, ...)
       │
       ├─> Create distributions (polymorphers)
       │   └─> make_block_matmul_a_m_k_distribution<BlockSize, M, K, KPack>()
       │       ├─> poly::merge (threads → partition)
       │       ├─> poly::split (M → M0 × M1)
       │       ├─> poly::split (K → K0 × K1)
       │       └─> poly::make_polymorpher(morphers, lengths)
       │
       ├─> Create element layouts
       │   └─> make_aliased_naive_packed_tensor_descriptor(element_dims, element_lengths)
       │
       ├─> Create tensor views
       │   └─> make_tensor_view(descriptor, memory_view)
       │
       ├─> Create distributed windows (ONE CALL)
       │   └─> make_distributed_window(
       │         tensor_view,                    ← Tensor
       │         {m_block * MPerBlock, 0},       ← Position
       │         constant<distribution>{},       ← Thread distribution
       │         constant<element_layout>{},     ← Per-thread layout
       │         constant<this_thread_block>{})  ← Partition scope
       │
       └─> Main Loop:
           ├─> masked_load_no_shuffle_vectorized (GMEM → VGPR)
           ├─> masked_store_no_shuffle_vectorized (VGPR → LDS)
           ├─> block_sync_lds()
           ├─> masked_load_no_shuffle_vectorized (LDS → VGPR)
           ├─> matmul_xdl (MFMA)
           └─> move_window (advance window position)
```

### Single-Window Creation

```
Distributed Window (Direct Creation)
  │
  ├─> Tensor Descriptor (global shape)
  │     make_aliased_naive_packed_tensor_descriptor(aliases<"M", "K">, {M, K})
  │
  ├─> Distribution (polymorpher)
  │     make_block_matmul_a_m_k_distribution()
  │       ├─> Thread partition: tid_m[32] × tid_k[8]
  │       ├─> M split: M[128] → M0[4] × M1[32]
  │       └─> K split: K[64] → K0[8] × K1[8]
  │
  ├─> Element Layout (per-thread data)
  │     [M0=4, K1=8] = 32 elements per thread
  │
  └─> make_distributed_window (ONE CALL)
        ├─> Position: {m_block * MPerBlock, 0}
        ├─> Distribution: explicit polymorpher
        └─> Returns: window with position + thread mapping
```

---

## Key Architectural Differences

### CK_Tile: Two-Window Architecture

**Design:** Hierarchical separation of concerns

```
Kernel Layer (WHAT to compute)
  ↓ Block windows
Pipeline Layer (HOW to access)
  ↓ Copy windows with thread distribution
Hardware Layer (MFMA/LDS)
```

**Benefits:**
- ✅ Swappable pipelines (compute-optimized, memory-optimized, async)
- ✅ Reusable kernel across problem sizes
- ✅ Auto-tuning by changing policies
- ✅ AMD-optimized (MFMA, LDS banking, wave coordination)

**Trade-offs:**
- ❌ More code navigation between layers
- ❌ ROCm-specific (not portable to CUDA)
- ❌ Steeper learning curve for full stack

### MINT: Single-Window Architecture

**Design:** Compositional polymorpher algebra

```
Polymorpher Algebra (transformation)
  ↓
Distributed Tensor (position + distribution)
  ↓
Platform Backend (ROCm or CUDA)
```

**Benefits:**
- ✅ Portable (ROCm + CUDA)
- ✅ Mathematical elegance
- ✅ Single abstraction for full mapping
- ✅ Concise kernel code

**Trade-offs:**
- ❌ Fixed distribution (less flexibility)
- ❌ Manual optimization required
- ❌ Steeper learning curve for polymorpher algebra
- ❌ Not as tuned as CK_Tile for AMD hardware

---

## Thread Distribution Comparison

### CK_Tile: Pattern-Based Automatic

```cpp
// Policy defines pattern
using TileEncodingPattern = tile_distribution_encoding_pattern_2d<
    BlockSize,        // 256 threads
    MPerBlock,        // 128
    KPerBlock,        // 64
    VecLoadSize,      // 8 (automatic or configured)
    tile_distribution_pattern::thread_raked,  // Pattern choice
    NumWaveGroups>;

auto distribution = TileEncodingPattern::make_2d_static_tile_distribution();
```

**Pattern: thread_raked**
- Optimizes for coalesced memory access in contiguous dimension
- Threads distributed to maximize vectorization
- Hardware-aware (wave size, LDS banks)

**Visual (simplified):**
```
128 ┌───────────────────────────┐
    │ T0→→ T1→→ T2→→ T3→→     │
M   │ T32→→ T33→→ ...         │
    │ T64→→ T65→→ ...         │
  0 └───────────────────────────┘
    0         K              64

Each → = vector of VecLoadSize elements
```

### MINT: Polymorpher Explicit

```cpp
// Explicit thread partitioning
constexpr auto K0 = kKPerBlock / kKPack;  // 64/8 = 8
constexpr auto tid_k = K0;                // 8 threads in K
constexpr auto tid_m = BlockSize / tid_k; // 256/8 = 32 threads in M
constexpr auto M0 = kMPerBlock / tid_m;   // 128/32 = 4 groups

// Build polymorpher
constexpr auto p_merge = poly::merge<nd_index<2>>{{tid_m, tid_k}};
constexpr auto m_split = poly::split<nd_index<2>>{{M0, tid_m}};
constexpr auto k_split = poly::split<nd_index<2>>{{K0, kKPack}};
```

**Pattern: M-strided groups**
- Each thread handles M0 groups vertically
- Fixed K-partition horizontally
- Explicit mathematical mapping

**Visual:**
```
128 ┌───────────────────────────┐
    │ T0 T1 T2 ... T7         │  M0=0 (rows 0-31)
 96 ├───────────────────────────┤
    │ T0 T1 T2 ... T7         │  M0=1 (rows 32-63)
 64 ├───────────────────────────┤
    │ T0 T1 T2 ... T7         │  M0=2 (rows 64-95)
 32 ├───────────────────────────┤
    │ T0 T1 T2 ... T7         │  M0=3 (rows 96-127)
  0 └───────────────────────────┘
    0         K              64

Each thread: 4 M-groups × 8 K-elements
```

---

## Working with the Code

### Understanding CK_Tile

**When reading CK_Tile code, follow this mental model:**

1. **Start at kernel layer** - understand the problem (tile sizes, data types)
2. **Identify block windows** - see what logical tiles are being processed
3. **Jump to pipeline** - see how windows are refined with thread distributions
4. **Check policy** - understand the optimization choices (patterns, vectorization)
5. **Look at pipeline operator** - see the data flow (GMEM→VGPR→LDS→MFMA)

**Common patterns:**
- `make_tile_window()` without distribution = block window (kernel layer)
- `make_tile_window()` with distribution = copy window (pipeline layer)
- `load_tile()` / `store_tile()` = data movement
- `block_sync_lds()` = synchronization point
- `BlockGemm()` = MFMA computation

### Understanding MINT

**When reading MINT code, follow this mental model:**

1. **Find distribution functions** - understand polymorpher composition
2. **Trace polymorpher algebra** - merge, split, rotate transformations
3. **Find window creation** - see position + distribution in one call
4. **Look at element layout** - understand per-thread data structure
5. **Follow main loop** - see load, store, compute pattern

**Common patterns:**
- `poly::merge()` = combine dimensions (threads → partition)
- `poly::split()` = split dimensions (M → M0 × M1)
- `make_distributed_window()` = position + distribution + element layout
- `masked_load_no_shuffle_vectorized()` = load with vectorization
- `matmul_xdl()` = MFMA computation

### Debugging Tips

**CK_Tile:**
- Check window sizes match tile sizes (MPerBlock, KPerBlock)
- Verify thread distribution creates expected elements per thread
- Look for sync points - missing sync causes race conditions
- Check LDS size calculation in GetSmemSize()
- Verify vector sizes are valid for data types

**MINT:**
- Verify polymorpher dimensions multiply correctly
- Check element_ndim() matches actual per-thread data
- Ensure freezed_dims align with descriptor aliases
- Verify vector_dims and vector_lengths are consistent
- Check thread partition matches block size

---

## Build and Test

### Build Examples

```bash
cd /home/khuagarw1/repos/rocm-libraries/projects/composablekernel

# Build CK_Tile example
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DGPU_TARGETS="gfx942" \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make example_universal_gemm -j32

# Run
./bin/example_universal_gemm

# Build MINT example
make example_rocm_gemm_xdl_v1 -j32

# Run
./bin/example_rocm_gemm_xdl_v1
```

### Test Matrix

Both implementations support:
- Data types: fp16, bf16, fp32, fp64, fp8, int8
- Layouts: RowMajor, ColumnMajor (A, B, C independently)
- Tile sizes: Configurable MPerBlock, NPerBlock, KPerBlock
- Block sizes: 64, 128, 256 threads
- GPUs: gfx908, gfx90a, gfx942, gfx950

---

## Common Tasks

### Task 1: Add New Tile Size to CK_Tile

1. **Modify problem configuration** (in kernel instantiation):
```cpp
constexpr index_t MPerBlock = 256;  // Change from 128
constexpr index_t NPerBlock = 256;
constexpr index_t KPerBlock = 32;   // Change from 64
```

2. **Verify LDS size** is within limits (policy.hpp):
```cpp
// Check GetSmemSize() output
constexpr index_t smem_size_a = ...;
constexpr index_t smem_size_b = ...;
static_assert(smem_size_a + smem_size_b <= 65536);  // LDS limit
```

3. **Rebuild and test**

### Task 2: Add New Tile Size to MINT

1. **Modify kernel template parameters**:
```cpp
constexpr index_t kMPerBlock = 256;  // Change
constexpr index_t kNPerBlock = 256;
constexpr index_t kKPerBlock = 32;   // Change
```

2. **Verify thread distribution** makes sense:
```cpp
constexpr auto K0 = kKPerBlock / kKPack;  // Must divide evenly
constexpr auto tid_k = K0;
constexpr auto tid_m = BlockSize / tid_k; // Must divide evenly
```

3. **Rebuild and test**

### Task 3: Compare Performance

```bash
# CK_Tile
./bin/example_universal_gemm

# MINT
./bin/example_rocm_gemm_xdl_v1

# Expected: CK_Tile ~10-20% faster due to AMD-specific optimizations
```

### Task 4: Port MINT Kernel to CUDA

MINT is designed for portability. Key changes needed:
1. Replace HIP intrinsics with CUDA equivalents
2. Change `__builtin_amdgcn_*` to `__nvvm_*`
3. Adjust wave size (64 → 32)
4. Use CUDA kernel launch syntax

---

## Further Reading

- **Architecture Comparison:** [`TILE_WINDOW_ARCHITECTURE.md`](~/repos/rocm-libraries/projects/composablekernel/TILE_WINDOW_ARCHITECTURE.md)
- **CK Terminology:** [`TERMINOLOGY.md`](~/repos/rocm-libraries/projects/composablekernel/TERMINOLOGY.md)
- **CK Main Guide:** [`CLAUDE.md`](~/repos/rocm-libraries/projects/composablekernel/CLAUDE.md)
- **MINT Overview:** [`include/mint/tile/tile_overview.md`](~/repos/rocm-libraries/projects/composablekernel/include/mint/tile/tile_overview.md)

---

## Quick Reference

### File Paths

```bash
# CK_Tile GEMM
KERNEL="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
PIPELINE="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
BASE="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
POLICY="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
BLOCK="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
EXAMPLE="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/example/ck_tile/03_gemm/universal_gemm.cpp"

# MINT
MINT_EXAMPLE="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp"
MINT_INCLUDE="/home/khuagarw1/repos/rocm-libraries/projects/composablekernel/include/mint"
```

### Key Concepts

| Concept | CK_Tile | MINT |
|---------|---------|------|
| **Window** | Two-phase (block → copy) | Single-phase (distributed) |
| **Distribution** | Pattern-based automatic | Polymorpher explicit |
| **Thread Mapping** | thread_raked/striped | M-group striding |
| **Abstraction** | Hierarchical layers | Compositional algebra |
| **Optimization** | AMD-specific | Portable baseline |
| **Complexity** | In separation | In composition |

---

**Last Updated:** 2026-03-24
**Author:** Technical Documentation
**Status:** Production Ready
