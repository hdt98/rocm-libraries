# Tile Window Architecture: CK_Tile vs MINT

**Author:** Technical Analysis
**Date:** 2026-03-24
**Context:** Composable Kernel GEMM Pipeline Design

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [CK_Tile Two-Window Architecture](#ck_tile-two-window-architecture)
3. [MINT Single-Window Architecture](#mint-single-window-architecture)
4. [Window Size Comparison](#window-size-comparison)
5. [Thread Distribution Comparison](#thread-distribution-comparison)
6. [Design Philosophy Analysis](#design-philosophy-analysis)
7. [Use Case Recommendations](#use-case-recommendations)
8. [Code Examples](#code-examples)
9. [References](#references)

---

## Executive Summary

This document analyzes two different approaches to tile-based GEMM kernel programming:

- **CK_Tile**: AMD's hierarchical tile programming model with two-window abstraction
- **MINT**: Metal INterface Tiles - portable polymorpher-based single-window model

### Key Findings

| Aspect | CK_Tile | MINT |
|--------|---------|------|
| **Window Abstraction** | Two-phase (Block → Copy) | Single-phase (Distributed) |
| **Complexity Location** | Hierarchical layers | Compositional algebra |
| **Portability** | ROCm-specific | ROCm + CUDA |
| **Policy Flexibility** | High (swappable pipelines) | Lower (fixed polymorphers) |
| **Learning Curve** | Moderate (layer navigation) | Steep (polymorpher algebra) |
| **Performance** | Highly optimized for AMD | Portable baseline |

---

## CK_Tile Two-Window Architecture

### Overview

CK_Tile implements a **separation of concerns** design where tile windowing happens in two distinct phases:

```
┌─────────────────────────────────────────┐
│  KERNEL LAYER                           │
│  - Workgroup positioning                │
│  - Block-level logical tiling           │
│  - Creates: Block Window                │
└──────────────┬──────────────────────────┘
               │ passes to pipeline
               ▼
┌─────────────────────────────────────────┐
│  PIPELINE LAYER                         │
│  - Thread-level memory access patterns  │
│  - Multiple specialized copy windows    │
│  - Creates: DRAM Copy, LDS Copy, LDS    │
│            Load windows                 │
└─────────────────────────────────────────┘
```

### Phase 1: Block Window Creation

**Location:** [`universal_gemm_kernel.hpp:820-894`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp#L820-L894)

```cpp
// Step 1: Create tensor view from global memory
const auto& as_tensor_view = generate_tuple([&](auto i) {
    using AiDataType = remove_cvref_t<std::tuple_element_t<i.value, AsDataType>>;
    return make_tensor_view<address_space_enum::global>(
        static_cast<const AiDataType*>(as_ptr[i]), as_desc[i]);
}, number<NumATensor>{});

// Step 2: Pad tensor view for boundary handling
const auto& as_pad_view = generate_tuple([&](auto i) {
    using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
    if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>) {
        return pad_tensor_view(as_tensor_view[i],
                              make_tuple(number<TilePartitioner::MPerBlock>{},
                                       number<TilePartitioner::KPerBlock>{}),
                              sequence<false, GemmPipeline::kPadK>{});
    }
    // ... ColumnMajor case
}, number<NumATensor>{});

// Step 3: Create block-level tile window (NO thread distribution)
const auto& as_block_window = generate_tuple([&](auto i) {
    using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
    if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>) {
        return make_tile_window(as_pad_view[i],
                               make_tuple(number<TilePartitioner::MPerBlock>{},
                                        number<TilePartitioner::KPerBlock>{}),
                               {i_m, 0});  // ← Workgroup coordinates
    }
    // ... ColumnMajor case
}, number<NumATensor>{});
```

**Characteristics:**
- ✅ Defines **which** tile this workgroup processes
- ✅ Positioned at workgroup coordinates `(i_m, i_n)`
- ✅ Handles boundary padding
- ❌ **NO thread-level distribution** - pure logical view

### Phase 2: Copy Window Creation

**Location:** [`gemm_pipeline_ag_bg_cr_base.hpp:152-291`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp#L152-L291)

```cpp
// Called from pipeline with the block window
auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                 const ALdsTensorView& a_lds_block_view,
                 const ALdsLoadTileDistr& a_lds_load_tile_distr,
                 const array<index_t, 2>& offset = {0, 0}) const {

    // Create DRAM copy window WITH thread distribution
    auto a_copy_dram_window = make_tile_window(
        dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(YPerTile{}, XPerTile{}),
        dram_block_window_tmp.get_window_origin() + offset,
        Policy::template MakeADramTileDistribution<Problem>());  // ← THREAD DISTRIBUTION

    // Create LDS store window WITH thread distribution
    auto a_copy_lds_window = make_tile_window(
        a_lds_block_view, a_lds_shape, {0, 0});

    // Create LDS load window WITH thread distribution (for GEMM)
    auto a_lds_gemm_window = make_tile_window(
        a_lds_block_view, a_lds_shape, {0, 0}, a_lds_load_tile_distr);

    return make_tuple(std::move(a_copy_dram_window),
                      std::move(a_copy_lds_window),
                      std::move(a_lds_gemm_window));
}
```

**From ONE block window, creates THREE specialized windows:**

1. **DRAM Copy Window**: Thread pattern for global memory loads
2. **LDS Store Window**: Thread pattern for LDS writes
3. **LDS Load Window**: Thread pattern for LDS reads (different from store!)

### Thread Distribution Creation

**Location:** [`gemm_universal_pipeline_ag_bg_cr_policy.hpp:714-748`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp#L714-L748)

```cpp
template <typename Problem>
CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution() {
    constexpr index_t BlockSize = Problem::kBlockSize;
    constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
    constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();
    constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    // Automatic distribution based on pattern
    using TileEncodingPattern = tile_distribution_encoding_pattern_2d<
        BlockSize,        // Total threads
        MPerBlock,        // Tile Y dimension
        KPerBlock,        // Tile X dimension
        VecLoadSize,      // Vectorization size
        getATileAccessPattern(),  // thread_raked or thread_striped
        NumWaveGroups>;   // Wave coordination

    return TileEncodingPattern::make_2d_static_tile_distribution();
}
```

**Key Features:**
- Pattern-based automatic distribution (`thread_raked`, `thread_striped`)
- Optimizes for memory coalescing
- Hardware-aware (wave size, LDS banks)
- Maximizes vector loads

### Why Two Windows?

#### Reason 1: Multiple Access Patterns from Same Logical Tile

```cpp
// Pipeline code creates 3 windows from 1 block window
auto&& [a_copy_dram_window,    // ← Pattern for GMEM→VGPR
        a_copy_lds_window,      // ← Pattern for VGPR→LDS
        a_lds_gemm_window] =    // ← Pattern for LDS→MFMA
    Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
```

**Same logical tile, THREE different thread access patterns:**
- DRAM load: Coalesced vector loads
- LDS store: Bank conflict avoidance
- LDS load: MFMA instruction requirements

#### Reason 2: Policy Separation

```
┌────────────────────────────────────┐
│ Problem Definition (WHAT)          │
│ - Tile sizes: MPerBlock, KPerBlock │
│ - Data types: ADataType, BDataType │
│ - Block window positioning         │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│ Policy Definition (HOW)            │
│ - Thread distribution patterns     │
│ - Vectorization strategies         │
│ - Memory access optimizations      │
└────────────────────────────────────┘
```

**Benefits:**
- Swap pipelines without changing kernel
- Auto-tune by changing policy
- Kernel reusable across architectures

#### Reason 3: Pipeline Flexibility

Same kernel works with different pipelines:

```cpp
// Pipeline Variant 1: Compute-optimized (V3)
using Pipeline1 = GemmPipelineAgBgCrCompV3<Problem, Policy>;

// Pipeline Variant 2: Memory-optimized (V2)
using Pipeline2 = GemmPipelineAgBgCrV2<Problem, Policy>;

// Pipeline Variant 3: Async copy
using Pipeline3 = GemmPipelineAgBgCrAsync<Problem, Policy>;

// SAME kernel code works with all three!
```

---

## MINT Single-Window Architecture

### Overview

MINT uses **polymorpher composition** to encode both positioning and distribution in a single abstraction:

```
┌─────────────────────────────────────────┐
│  Tensor Descriptor                      │
│  - Global tensor shape                  │
│  - Memory layout                        │
└──────────────┬──────────────────────────┘
               │
               ├──► Polymorpher (transformation algebra)
               │    - Merge threads → partition
               │    - Split dimensions → sub-tiles
               │    - Encode access pattern
               │
               ▼
┌─────────────────────────────────────────┐
│  Distributed Window                     │
│  - Position + Distribution in ONE       │
│  - Element layout (per-thread data)     │
│  - Thread partition scope               │
└─────────────────────────────────────────┘
```

### Polymorpher Construction

**Location:** [`rocm_gemm_xdl_v1.cpp:164-211`](~/repos/rocm-libraries/projects/composablekernel/example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp#L164-L211)

```cpp
template <index_t BlockSize, index_t kMPerBlock, index_t kKPerBlock, index_t kKPack>
__device__ constexpr auto make_block_matmul_a_m_k_distribution() {
    // Step 1: Calculate thread partitioning
    constexpr auto K0 = kKPerBlock / kKPack;    // 64/8 = 8
    constexpr auto tid_k = K0;                  // 8 threads in K dim
    constexpr auto tid_m = BlockSize / tid_k;   // 256/8 = 32 threads in M dim
    constexpr auto M1 = tid_m;                  // 32
    constexpr auto M0 = kMPerBlock / M1;        // 128/32 = 4

    // Step 2: Define transformation primitives
    constexpr auto p_merge = poly::merge<nd_index<2>>{{tid_m, tid_k}};
    constexpr auto m_split = poly::split<nd_index<2>>{{M0, M1}};
    constexpr auto k_split = poly::split<nd_index<2>>{{K0, kKPack}};

    // Step 3: Define dimension connections
    constexpr auto dim_pairs = nd_array<index_t, 2, 2, 2>{
        {{0, 1}, {1, 1}},  // Partition merges to M dimension
        {{0, 2}, {2, 0}}   // Partition merges to K dimension
    };

    // Step 4: Build morphers tuple
    constexpr auto morphers = mint::make_tuple(p_merge, m_split, k_split);

    // Step 5: Create dimension aliases
    constexpr auto alias_to_morpher = []() {
        static_map<alias_t, index_t, 3> ret;
        ret["P"] = 0;  // Partition
        ret["M"] = 1;  // M dimension
        ret["K"] = 2;  // K dimension
        return ret;
    }();

    constexpr auto alias_to_dim = []() {
        static_map<alias_t, nd_index<2>, 7> ret;
        ret["P"] = {0, 0};      // Partition (flat)
        ret["M"] = {1, 2};      // M (split)
        ret["M_0"] = {1, 0};    // M outer
        ret["M_1"] = {1, 1};    // M inner
        ret["K"] = {2, 2};      // K (split)
        ret["K_0"] = {2, 0};    // K outer
        ret["K_1"] = {2, 1};    // K inner (pack)
        return ret;
    }();

    // Step 6: Define dimension lengths
    constexpr auto lengths = nd_index<7>{
        BlockSize,    // P: 256 threads
        kMPerBlock,   // M: 128
        M0,           // M_0: 4
        M1,           // M_1: 32
        kKPerBlock,   // K: 64
        K0,           // K_0: 8
        kKPack        // K_1: 8
    };

    // Step 7: Compose into polymorpher
    constexpr auto poly = poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
        morphers, lengths);

    // Step 8: Return distributed tensor descriptor
    constexpr auto top_dim_aliases = array<alias_t, 2>{"M", "K"};
    constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
    constexpr auto element_dim_aliases = array<alias_t, 2>{"M_0", "K_1"};

    return distributed_tensor_descriptor<
        poly,
        top_dim_aliases,
        partition_dim_aliases,
        element_dim_aliases>{};
}
```

### What This Polymorpher Encodes

The single polymorpher contains:

```
Logical Space:        Thread Space:         Element Space:
   M[128]                P[256]              M_0[4]
     ↓                     ↓                   ↓
  M_0[4] × M_1[32]    tid_m[32] × tid_k[8]  K_1[8]

   K[64]
     ↓
  K_0[8] × K_1[8]

Mapping:
  Thread (tid_m, tid_k) owns elements at:
    M positions: [M_0 * tid_m + m0] for m0 ∈ [0, M_0)
    K positions: [K_0 * tid_k + k1] for k1 ∈ [0, K_1)
```

### Single Window Creation

**Location:** [`rocm_gemm_xdl_v1.cpp:756-761`](~/repos/rocm-libraries/projects/composablekernel/example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp#L756-L761)

```cpp
// Create global tensor descriptor
const auto a_gmem_tensor_desc = make_aliased_naive_packed_tensor_descriptor(
    aliases<"M", "K">{}, alias<"Offset">{}, {m_size, k_size});

// Create distribution
constexpr auto a_block_copy_dstr = make_block_matmul_a_m_k_distribution<
    kThreadPerBlock, kMPerBlock, kKPerBlock, kKPack>();

// Create element layout
constexpr auto a_block_copy_element_layout =
    make_aliased_naive_packed_tensor_descriptor(
        make_index_sequence<a_block_copy_dstr.element_ndim()>{},
        index_constant<-1>{},
        a_block_copy_dstr.element_lengths());  // [4, 8]

// Create tensor view
const auto a_gmem_tensor_view = make_tensor_view(
    a_gmem_tensor_desc,
    make_global_memory_view(p_a_gmem, a_gmem_tensor_desc.bottom_lengths()[0]));

// ONE CALL: Create distributed window with EVERYTHING
auto a_block_copy_gmem_window = make_distributed_window(
    a_gmem_tensor_view,              // ← Tensor view
    {m_block * kMPerBlock, 0},       // ← Block positioning
    constant<a_block_copy_dstr>{},   // ← Thread distribution
    constant<a_block_copy_element_layout>{},  // ← Per-thread layout
    constant<this_thread_block{}>{});         // ← Thread partition
```

**One window contains:**
- Global tensor shape
- Block positioning (`m_block * kMPerBlock`)
- Thread distribution (polymorpher)
- Element layout (per-thread data structure)

### Multiple Windows in MINT

MINT **DOES** create multiple windows, but directly:

```cpp
// GMEM window for block copy
auto a_block_copy_gmem_window = make_distributed_window(
    a_gmem_tensor_view,
    {m_block * kMPerBlock, 0},
    constant<a_block_copy_dstr>{},     // ← Block-level distribution
    constant<a_block_copy_element_layout>{},
    constant<this_thread_block{}>{});

// SMEM window for block copy (same distribution, different tensor)
auto a_block_copy_smem_window = make_distributed_window(
    a_smem_tensor_view,
    {0, 0},
    constant<a_block_copy_dstr>{},     // ← SAME distribution
    constant<a_block_copy_element_layout>{},
    constant<this_thread_block{}>{});

// SMEM window for warp matmul (DIFFERENT distribution)
auto a_warp_matmul_smem_window = make_distributed_window(
    a_smem_tensor_view,
    {m_warp * kMPerWarp * kMRepeat, 0},
    constant<a_warp_matmul_dstr>{},    // ← DIFFERENT distribution
    constant<a_warp_matmul_elem_layout>{},
    constant<thread_in_this_warp{}>{});
```

**Key difference:** No intermediate "block window" - windows created directly from descriptors.

---

## Window Size Comparison

### Configuration

**CK_Tile Example:**
```cpp
// Typical configuration
constexpr index_t MPerBlock = 128;
constexpr index_t NPerBlock = 256;
constexpr index_t KPerBlock = 64;
constexpr index_t BlockSize = 256;
constexpr index_t VectorSizeA = 8;
```

**MINT Example:**
```cpp
// rocm_gemm_xdl_v1.cpp configuration
constexpr index_t kMPerBlock = 128;
constexpr index_t kNPerBlock = 256;
constexpr index_t kKPerBlock = 64;
constexpr index_t kThreadPerBlock = 256;
constexpr index_t kKPack = 8;  // Vectorization
```

### Window Dimensions

| Tensor | CK_Tile | MINT | Identical? |
|--------|---------|------|------------|
| **A (M×K)** | 128 × 64 | 128 × 64 | ✅ YES |
| **B (N×K)** | 256 × 64 | 256 × 64 | ✅ YES |
| **C (M×N)** | 128 × 256 | 128 × 256 | ✅ YES |
| **Total Elements** | 8,192 | 8,192 | ✅ YES |
| **Elements/Thread** | 32 | 32 | ✅ YES |
| **Threads** | 256 | 256 | ✅ YES |

**Conclusion:** Window sizes are **configuration-dependent** and can be identical.

---

## Thread Distribution Comparison

### CK_Tile Thread Distribution

**Pattern:** `thread_raked` (configurable)

```cpp
// Automatic calculation by tile_distribution_encoding_pattern_2d
// Goal: Maximize coalesced access in contiguous dimension

For A[128, 64] with BlockSize=256, VecLoadSize=8:

Elements per thread = (128 * 64) / 256 = 32
Vectors per thread = 32 / 8 = 4

Typical raked distribution (simplified):
┌─────────────────────────────────────┐
│ T0→→ T1→→ T2→→ ... T31→→           │  Row 0-3
│ T32→→ T33→→ ...                    │  Row 4-7
│ ...                                │
│ T224→→ T225→→ ... T255→→           │  Row 124-127
└─────────────────────────────────────┘
     ← K dimension (64 elements) →

Each → represents a vector of 8 elements
Threads assigned in raked pattern for coalescing
```

**Characteristics:**
- Automatic optimization based on pattern
- Adapts to architecture (wave size, LDS banks)
- Prioritizes memory coalescing
- Implementation in `tile_distribution_encoding_pattern_2d`

### MINT Thread Distribution

**Pattern:** Explicit polymorpher-defined

```cpp
// Explicit thread assignment
constexpr auto tid_k = 8;   // 8 threads in K dimension
constexpr auto tid_m = 32;  // 32 threads in M dimension
constexpr auto M0 = 4;      // 4 M groups per thread
constexpr auto K1 = 8;      // 8 K elements per thread (vectorization)

Distribution:
┌─────────────────────────────────────┐
│ T0  T1  T2  ... T7  (8 threads/row) │  M0=0 (rows 0-31)
│ T8  T9  T10 ... T15                │
│ ...                                │
│ T24 T25 T26 ... T31                │
├─────────────────────────────────────┤
│ T0  T1  T2  ... T7  (SAME threads) │  M0=1 (rows 32-63)
│ ...                                │
├─────────────────────────────────────┤
│ T0  T1  T2  ... T7                 │  M0=2 (rows 64-95)
│ ...                                │
├─────────────────────────────────────┤
│ T0  T1  T2  ... T7                 │  M0=3 (rows 96-127)
│ ...                                │
└─────────────────────────────────────┘

Thread (tid_m, tid_k) owns:
  M positions: [0*32+tid_m, 1*32+tid_m, 2*32+tid_m, 3*32+tid_m]
  K positions: [tid_k*8 + 0...7]
```

**Characteristics:**
- Explicit M-strided access pattern
- Fixed by polymorpher definition
- Each thread handles M0=4 groups vertically
- Partition: 32 threads in M, 8 threads in K

### Concrete Example: Thread 0

#### CK_Tile Thread 0 (thread_raked pattern)

```
Approximate assignment (pattern-dependent):
Position (M, K):
  (0, 0-7)    ← Vector 0
  (0, 8-15)   ← Vector 1
  (0, 16-23)  ← Vector 2
  (0, 24-31)  ← Vector 3

Total: 32 elements in 4 consecutive vectors along K dimension
```

#### MINT Thread 0

```
Explicit assignment:
Thread 0 = (tid_m=0, tid_k=0)

Position (M, K):
  (0, 0-7)    ← M0=0, K1=0-7
  (32, 0-7)   ← M0=1, K1=0-7
  (64, 0-7)   ← M0=2, K1=0-7
  (96, 0-7)   ← M0=3, K1=0-7

Total: 32 elements in 4 M-strided groups
```

### Visual Comparison

```
CK_Tile (thread_raked):
128 ┌──────────────────────────┐
    │ T0→→→→ T1→→→→ T2→→→→    │
M   │ T64→→→→ T65→→→→ ...     │
    │ ...                     │
  0 └──────────────────────────┘
    0            K          64

MINT (M-strided):
128 ┌──────────────────────────┐
    │ T0 T1 ... T7            │ M0=0
 96 ├──────────────────────────┤
    │ T0 T1 ... T7            │ M0=1
 64 ├──────────────────────────┤
    │ T0 T1 ... T7            │ M0=2
 32 ├──────────────────────────┤
    │ T0 T1 ... T7            │ M0=3
  0 └──────────────────────────┘
    0            K          64
```

### Comparison Summary

| Aspect | CK_Tile | MINT |
|--------|---------|------|
| **Formula** | Pattern-based (automatic) | Polymorpher-explicit |
| **Thread Assignment** | Raked/striped patterns | M-group striding |
| **Vectorization** | `VecLoadSize` parameter | `kKPack` parameter |
| **Optimization** | Architecture-aware | Fixed by definition |
| **Flexibility** | Swappable patterns | Fixed by polymorpher |
| **Coalescing** | Automatic | Manual design |

**Conclusion:** Different formulas, different spatial arrangements, but both achieve:
- ✅ Same total window size
- ✅ Same elements per thread
- ✅ Coalesced memory access

---

## Design Philosophy Analysis

### CK_Tile Philosophy: Hierarchical Separation

```
┌─────────────────────────────────────┐
│ KERNEL LAYER (Problem: WHAT)        │
│ - Defines logical computation       │
│ - Block-level positioning           │
│ - Hardware-agnostic                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ PIPELINE LAYER (Policy: HOW)        │
│ - Memory access strategies          │
│ - Thread distribution patterns      │
│ - Multiple specialized windows      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ HARDWARE LAYER                      │
│ - MFMA/XDL instructions             │
│ - LDS banking                       │
│ - Wave coordination                 │
└─────────────────────────────────────┘
```

**Advantages:**
- ✅ Clear separation of concerns
- ✅ Easy to swap pipelines for different GPUs
- ✅ Reusable kernel across problem sizes
- ✅ Auto-tuning by changing policies
- ✅ Domain experts can focus on one layer

**Disadvantages:**
- ❌ More code navigation required
- ❌ Indirection between layers
- ❌ AMD/ROCm-specific optimizations
- ❌ Steeper learning curve for full stack

### MINT Philosophy: Compositional Algebra

```
┌─────────────────────────────────────┐
│ POLYMORPHER ALGEBRA                 │
│ - Single mathematical transformation│
│ - Encodes: position + distribution  │
│ - Compose primitives:               │
│   • merge (threads → partition)     │
│   • split (dimension → sub-dims)    │
│   • rotate (bank conflict avoid)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ DISTRIBUTED TENSOR                  │
│ - Tensor view + polymorpher         │
│ - Platform-agnostic abstraction     │
│ - Compiles to ROCm or CUDA          │
└─────────────────────────────────────┘
```

**Advantages:**
- ✅ Portable (ROCm + CUDA)
- ✅ Mathematical elegance
- ✅ Single abstraction for full mapping
- ✅ Compile-time verification
- ✅ Concise kernel code

**Disadvantages:**
- ❌ Steep learning curve (polymorpher algebra)
- ❌ Less flexibility for specialized patterns
- ❌ Manual optimization required
- ❌ Not as tuned as CK_Tile for AMD hardware

### Trade-off Matrix

| Requirement | CK_Tile | MINT | Winner |
|-------------|---------|------|--------|
| **Performance on AMD GPUs** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | CK_Tile |
| **Cross-platform portability** | ⭐ | ⭐⭐⭐⭐⭐ | MINT |
| **Auto-tuning capability** | ⭐⭐⭐⭐⭐ | ⭐⭐ | CK_Tile |
| **Code maintainability** | ⭐⭐⭐ | ⭐⭐⭐⭐ | MINT |
| **Learning curve** | ⭐⭐⭐ | ⭐⭐ | CK_Tile |
| **Mathematical elegance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | MINT |
| **Kernel code conciseness** | ⭐⭐⭐ | ⭐⭐⭐⭐ | MINT |
| **Hardware specialization** | ⭐⭐⭐⭐⭐ | ⭐⭐ | CK_Tile |

---

## Use Case Recommendations

### Choose CK_Tile When:

1. **AMD GPU Performance is Critical**
   - Production ML training/inference
   - HPC applications on MI300X/MI250X
   - Need maximum hardware utilization

2. **Auto-Tuning Required**
   - Multiple problem sizes
   - Need kernel selection at runtime
   - Want automatic optimization

3. **AMD Ecosystem Integration**
   - Using ROCm stack exclusively
   - Need MFMA/XDL instructions
   - Integrating with rocBLAS/MIOpen

4. **Multiple Pipeline Variants**
   - Different memory access patterns
   - Compute-bound vs memory-bound
   - Experimental kernel optimization

### Choose MINT When:

1. **Cross-Platform Development**
   - Need same code on ROCm and CUDA
   - Research prototypes
   - Educational purposes

2. **Mathematical Clarity Preferred**
   - Want explicit transformation algebra
   - Academic publication
   - Formal verification

3. **Simpler Problem Space**
   - Single kernel variant
   - Fixed problem sizes
   - Don't need auto-tuning

4. **Learning Tile Programming**
   - Understand fundamentals
   - Build from first principles
   - Less framework magic

### Hybrid Approach (Recommended for Development)

```cpp
// Use MINT for kernel logic (portable)
__global__ void MyGemmKernel(...) {
    // MINT distributed windows
    auto window = make_distributed_window(...);

    // MINT tile operations
    warp::masked_load(window, ...);
}

// Use CK_Tile for infrastructure
int main() {
    // CK_Tile host utilities
    auto tensor_view = make_tensor_view(...);
    auto kargs = make_kernel_args(...);

    // CK_Tile kernel launch
    launch_kernel(MyGemmKernel, ...);

    // CK_Tile validation
    validate_result(...);
}
```

**Benefits:**
- Portable device kernels
- Mature host infrastructure
- Best of both worlds

---

## Code Examples

### Example 1: A Tensor Window Creation

#### CK_Tile

```cpp
// Step 1: Kernel layer creates block window
const auto a_tensor_view = make_tensor_view<address_space_enum::global>(
    p_a, make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, 1)));

const auto a_pad_view = pad_tensor_view(
    a_tensor_view,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    sequence<false, true>{});  // Pad K dimension

const auto a_block_window = make_tile_window(
    a_pad_view,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    {block_idx_m * MPerBlock, 0});  // Block positioning

// Step 2: Pipeline layer creates copy windows
auto GetAWindows(...) {
    // DRAM copy window
    auto a_copy_dram_window = make_tile_window(
        a_block_window.get_bottom_tensor_view(),
        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
        a_block_window.get_window_origin(),
        Policy::template MakeADramTileDistribution<Problem>());  // Thread distribution

    // LDS copy window
    auto a_copy_lds_window = make_tile_window(
        a_lds_view,
        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
        {0, 0});

    return make_tuple(a_copy_dram_window, a_copy_lds_window);
}
```

#### MINT

```cpp
// Single step: Create distributed window
const auto a_gmem_desc = make_aliased_naive_packed_tensor_descriptor(
    aliases<"M", "K">{}, alias<"Offset">{}, {M, K});

constexpr auto a_dstr = make_block_matmul_a_m_k_distribution<
    BlockSize, MPerBlock, KPerBlock, KPack>();

constexpr auto a_elem_layout = make_aliased_naive_packed_tensor_descriptor(
    make_index_sequence<a_dstr.element_ndim()>{},
    index_constant<-1>{},
    a_dstr.element_lengths());

const auto a_view = make_tensor_view(
    a_gmem_desc,
    make_global_memory_view(p_a, a_gmem_desc.bottom_lengths()[0]));

// ONE window with position + distribution
auto a_window = make_distributed_window(
    a_view,
    {block_idx_m * MPerBlock, 0},  // Position
    constant<a_dstr>{},            // Distribution
    constant<a_elem_layout>{},     // Element layout
    constant<this_thread_block{}>{});
```

### Example 2: Loading Data

#### CK_Tile

```cpp
// Load from GMEM to VGPR
auto a_tile = load_tile_with_elementwise(a_copy_dram_window, elementwise_func);

// Store from VGPR to LDS
store_tile(a_copy_lds_window, a_tile);

// Sync
block_sync_lds();

// Load from LDS to VGPR for MFMA
load_tile(a_mfma_tile, a_lds_gemm_window);
```

#### MINT

```cpp
// Load from GMEM to VGPR
constexpr auto vector_dims = array<alias_t, 1>{"K_1"};
constexpr auto vector_lengths = array<index_t, 1>{kKPack};

auto a_tile = mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
    vector_dims, vector_lengths, freezed_dims>(a_gmem_window, mask);

// Store from VGPR to LDS
mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
    vector_dims, vector_lengths, freezed_dims>(a_lds_window, mask, a_tile);

// Sync
block_sync_lds();

// Load from LDS for MFMA
auto a_mfma_tile = mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
    warp_vector_dims, warp_vector_lengths, warp_freezed_dims>(a_lds_window, mask);
```

### Example 3: Thread Distribution Definition

#### CK_Tile

```cpp
// Automatic distribution based on pattern
template <typename Problem>
static constexpr auto MakeADramTileDistribution() {
    constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();

    using TileEncodingPattern = tile_distribution_encoding_pattern_2d<
        Problem::kBlockSize,
        Problem::BlockGemmShape::kM,
        Problem::BlockGemmShape::kK,
        VecLoadSize,
        tile_distribution_pattern::thread_raked,  // Pattern choice
        Problem::NumWaveGroups>;

    return TileEncodingPattern::make_2d_static_tile_distribution();
}
```

#### MINT

```cpp
// Explicit polymorpher composition
template <index_t BlockSize, index_t M, index_t K, index_t KPack>
constexpr auto make_distribution() {
    constexpr auto K0 = K / KPack;
    constexpr auto tid_k = K0;
    constexpr auto tid_m = BlockSize / tid_k;
    constexpr auto M0 = M / tid_m;

    // Build polymorpher from primitives
    constexpr auto p_merge = poly::merge<nd_index<2>>{{tid_m, tid_k}};
    constexpr auto m_split = poly::split<nd_index<2>>{{M0, tid_m}};
    constexpr auto k_split = poly::split<nd_index<2>>{{K0, KPack}};

    constexpr auto dim_pairs = nd_array<index_t, 2, 2, 2>{
        {{0, 1}, {1, 1}}, {{0, 2}, {2, 0}}};

    constexpr auto morphers = mint::make_tuple(p_merge, m_split, k_split);
    // ... (alias definitions)

    constexpr auto poly = poly::make_polymorpher<dim_pairs, ...>(morphers, lengths);

    return distributed_tensor_descriptor<poly, ...>{};
}
```

---

## References

### CK_Tile Documentation

- **Universal GEMM Kernel:** [`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp)
- **Pipeline Base:** [`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp)
- **Pipeline V3:** [`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp)
- **Policy:** [`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`](~/repos/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp)

### MINT Documentation

- **GEMM XDL Example:** [`example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp`](~/repos/rocm-libraries/projects/composablekernel/example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp)
- **Polymorpher Overview:** `include/mint/poly/fundamental_morpher.md`
- **Distributed Tensors:** `include/mint/tensor/distributed_tensor.md`
- **Tile Overview:** `include/mint/tile/tile_overview.md`

### Key Concepts

- **Tensor Coordinate Transformation:** `TERMINOLOGY.md`
- **Problem vs Policy:** `CLAUDE.md`
- **MINT Framework:** `CLAUDE.md` (MINT section)

---

## Conclusion

Both CK_Tile and MINT achieve efficient tile-based GEMM kernels through different architectural philosophies:

**CK_Tile** prioritizes **performance and flexibility** through hierarchical separation:
- Two-window design separates positioning from distribution
- Policy-based optimization for AMD GPUs
- Maximum performance but ROCm-specific

**MINT** prioritizes **portability and elegance** through compositional algebra:
- Single-window design unifies positioning and distribution
- Polymorpher-based mathematical abstraction
- Cross-platform but requires manual tuning

The choice depends on your priorities:
- **Need maximum AMD performance?** → CK_Tile
- **Need cross-platform portability?** → MINT
- **Development/Learning?** → Hybrid approach

Both represent sophisticated approaches to GPU kernel programming, with different trade-offs in the complexity-flexibility-portability triangle.

---

**Document Version:** 1.0
**Last Updated:** 2026-03-24
**Tested On:** AMD MI300X (gfx942), ROCm 7.2.0
