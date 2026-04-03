# Gemm: Understanding the Kernel Entry Point

This document explains the `Gemm` struct, which serves as the **top-level kernel functor** for
our GEMM GPU kernel. We'll dive deep into how raw memory pointers are transformed into structured
tensor views and how the computation is dispatched through the hierarchy.

## Overview

The `Gemm` struct (defined in `practice_gemm.hpp`) is a templated kernel functor that:
1. Takes raw device memory pointers for matrices A, B, and C
2. Wraps them into **tensor views** — logical, structured views over physical memory
3. Maps thread blocks to output tiles and dispatches to `GridGemm` for computation

```cpp
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename CElementFunction,
          index_t kAAlignment,
          index_t kBAlignment,
          index_t kCAlignment,
          index_t kBlockSize_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          index_t kKPerBlock_>
struct Gemm
{
    // ... nested GridGemmPolicy ...

    CK_TILE_DEVICE void operator()(const ADataType* p_a,
                                   const BDataType* p_b,
                                   CDataType* p_c,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t Lda,
                                   const index_t Ldb,
                                   const index_t Ldc,
                                   const CElementFunction& c_element_func) const
    {
        // Step 1: Create tensor views over raw memory
        auto a_grid = make_naive_tensor_view<address_space_enum::global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), number<kAAlignment>{}, number<1>{});

        auto b_grid = make_naive_tensor_view<address_space_enum::global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), number<kBAlignment>{}, number<1>{});

        auto c_grid = make_naive_tensor_view<address_space_enum::global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), number<kCAlignment>{}, number<1>{});

        // Step 2: Dispatch to GridGemm
        GridGemm<GridGemmProblem_, GridGemmPolicy>{}(a_grid, b_grid, c_grid, c_element_func);
    }
};
```

---

## What are Tensor Views?

A **tensor view** is a **logical, structured view over raw physical memory**. It doesn't own or
allocate memory — it provides a way to interpret and access existing memory as a
multi-dimensional tensor, carrying shape, strides, and vectorization guarantees.

### Key Components of a Tensor View:

1. **Address space**: Where the data lives (`global`/DRAM, `lds`/shared, `vgpr`/registers)
2. **Raw pointer**: Points to the actual data in memory
3. **Shape (lengths)**: Dimensions of the tensor (e.g., M×K for matrix A)
4. **Strides**: How to navigate through memory to access elements
5. **Guaranteed vector length**: How many consecutive elements fit in one vector instruction
6. **Guaranteed vector stride**: The stride of those vectorizable elements

---

## The Memory Abstraction Hierarchy

CK Tile uses a three-layer abstraction to go from raw memory to structured tensors:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: TENSOR VIEW                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ • Logical multi-dimensional structure                   │ │
│ │ • Shape: (M, K)                                         │ │
│ │ • Strides: (Lda, 1) for row-major layout               │ │
│ │ • Provides: coordinate-based access, tile windows       │ │
│ │ • Knows: How to map (i,j) → linear offset              │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           ↓ wraps                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Layer 2: BUFFER VIEW                                    │ │
│ │ • Linear view of memory                                 │ │
│ │ • Pointer: p_data_ → device memory                      │ │
│ │ • Size: total number of elements                        │ │
│ │ • Address space: global / lds / generic                 │ │
│ │ • Provides: vectorized loads/stores, bounds checking    │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           ↓ wraps                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Layer 1: RAW PHYSICAL MEMORY                            │ │
│ │ [e₀][e₁][e₂]...[eN]    ← contiguous bytes in DRAM      │ │
│ │  ↑                                                      │ │
│ │  p_a (raw pointer from hipMalloc)                       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: `make_naive_tensor_view`

Let's break down the function call for matrix A:

```cpp
auto a_grid = make_naive_tensor_view<address_space_enum::global>(
    p_a,                         // Raw pointer to device memory
    make_tuple(M, K),            // Shape: M rows, K cols
    make_tuple(Lda, 1),          // Strides: Lda elements to the next row, 1 between cols
    number<kAAlignment>{},       // Guaranteed vector length (8 fp16 = 128 bits)
    number<1>{}                  // Guaranteed vector stride (cols are contiguous)
);
```

### Parameter Breakdown:

**1. `address_space_enum::global`**

Specifies where the memory lives:
- `global`: GPU global memory (DRAM) — slowest, largest
- `lds`: Local Data Share (LDS / shared memory) — fast, limited (~64KB per block)
- `vgpr`: Vector General Purpose Registers — fastest, smallest

**2. `p_a` — Raw Pointer**

The raw device memory pointer (from `hipMalloc` via `DeviceMem`). Points to the start of the
matrix data.

**3. `make_tuple(M, K)` — Shape**

Defines the logical dimensions: M rows, K columns. This is the logical view, independent
of physical layout.

**4. `make_tuple(Lda, 1)` — Strides**

Defines how to navigate through memory:
- **Dim 0 (rows)**: stride = `Lda` = K — to move to the next row, skip K elements
- **Dim 1 (cols)**: stride = `1` — consecutive columns are contiguous

For B (stored as `[N, K]`): shape is `(N, K)`, strides are `(Ldb, 1)`. N is the leading
dimension, K is contiguous — exactly matching B's `[N, K]` memory layout.

**5. `number<kAAlignment>{}` — Guaranteed Vector Length**

`kAAlignment = 8` tells the system: "The last dimension (K) is guaranteed to have at least 8
consecutive fp16 elements aligned in memory — safe to issue one `global_load_dwordx4`."

This enables the compiler to emit vectorized loads without runtime alignment checks:
```cpp
// What the compiler produces:
buffer_load_dwordx4 v[0:3], ...   // loads 8 fp16 = 128 bits in one instruction
```

**6. `number<1>{}` — Guaranteed Vector Stride**

Confirms that consecutive elements in the K dimension have stride 1 (contiguous in memory).
A stride > 1 would prevent vectorization.

---

## B's `[N, K]` Tensor View

Matrix B is stored in `[N, K]` layout (not `[K, N]`):

```cpp
auto b_grid = make_naive_tensor_view<address_space_enum::global>(
    p_b,
    make_tuple(N, K),    // N rows, K cols  ← N is leading dimension
    make_tuple(Ldb, 1),  // stride between N rows = Ldb = K; K cols are contiguous
    number<kBAlignment>{},
    number<1>{});
```

This layout means "row i of B in memory" = the K-vector `B[i, 0:K-1]`. When the kernel
creates a tile window `{iN, 0}` of size `(kNPerBlock, kKPerBlock)`, it reads `kNPerBlock`
consecutive rows of B, each being a K-slice — the coalesced pattern the kernel relies on.

---

## `GridGemmProblem_`: The Type-Tag Pattern

```cpp
// Inside Gemm:
using GridGemmProblem_ =
    GridGemmProblem<ADataType, BDataType, AccDataType, CDataType, CElementFunction>;
```

`GridGemmProblem` is an **empty struct with type aliases** — a type-tag. No data, no
member functions. Its sole purpose is to carry all data-type information as a single
template parameter through the hierarchy:

```
Gemm<ADataType, BDataType, AccDataType, CDataType, CElementFunction, ...>
    → GridGemmProblem_  (type tag bundles all data types)
        → GridGemm<GridGemmProblem_, GridGemmPolicy>
            → BlockGemmPipelineAGmemBGmemCReg<BlockGemmPipelineProblem_>
                → BlockGemmASmemBSmemCReg<Problem>
```

Without this pattern, every level of the hierarchy would need to carry `ADataType, BDataType,
AccDataType, CDataType, CElementFunction` as separate template parameters — much more verbose.

---

## `GridGemmPolicy`: Nested Policy Struct

```cpp
struct GridGemmPolicy
{
    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kMPerBlock = kMPerBlock_;
    static constexpr index_t kNPerBlock = kNPerBlock_;
    static constexpr index_t kKPerBlock = kKPerBlock_;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0);

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemmPipeline();
};
```

`GridGemmPolicy` is a **nested struct** inside `Gemm`. It captures the tile dimensions
(`kMPerBlock`, `kNPerBlock`, `kKPerBlock`) that were passed as template parameters to `Gemm`,
making them accessible to `GridGemm` via the Policy type.

**Problem vs. Policy:**
- `Problem` = **what**: data types, shapes — describes the mathematical problem
- `Policy` = **how**: tile mapping, pipeline selection — describes how to partition and execute

Separating these allows swapping execution strategies (different LDS layouts, different
pipeline types) without changing the Problem definition.

---

## Dispatch to GridGemm

```cpp
GridGemm<GridGemmProblem_, GridGemmPolicy>{}(a_grid, b_grid, c_grid, c_element_func);
```

`GridGemm` (in `host_level/grid_gemm.hpp`) is the device-side dispatcher that:
1. Extracts M, N, K from the tensor descriptors
2. Converts the linear `block_id` to a 2D tile coordinate `(iM, iN)` using `GridGemmPolicy::MakeBlock2TileMap`
3. Creates `a_block_window` at origin `(iM, 0)` and `b_block_window` at origin `(iN, 0)`
4. Calls `BlockGemmPipelineAGmemBGmemCReg` to compute the C tile
5. Applies `CElementFunction` and stores to C

---

## Visual Example: Matrix A Memory Layout

Let's visualize how matrix A (M×K = 3328×4096, fp16) is organized:

### Raw Physical Memory (Linear):
```
GPU DRAM:
│ a[0,0] │ a[0,1] │ ... │ a[0,4095] │ a[1,0] │ a[1,1] │ ... │ a[3327,4095] │
↑                                    ↑
Row 0 (K=4096 elements)             Row 1 (K=4096 elements)

Total: 3328 × 4096 × 2 bytes/fp16 = ~27 MB
```

### Tensor View Layer:
```
tensor_view: shape=(3328, 4096), strides=(4096, 1), vec_len=8
┌──────────────────────────────────────────────────┐
│ Logical 2D View:                                  │
│   Col: 0  1  2 ... 4095                          │
│ R0: [a₀₀][a₀₁]...[a₀,₄₀₉₅] ← can load 8 at once │
│ R1: [a₁₀][a₁₁]...[a₁,₄₀₉₅]                       │
│ ...                                               │
│ R3327: [...]                                      │
│                                                   │
│ Provides: make_tile_window(), vectorized loads    │
└──────────────────────────────────────────────────┘
```

---

## Complete Transformation: Raw Memory → Tensor View

```
Step 1: Kernel launch (host side)
    hipMalloc → raw pointer p_a

Step 2: Inside kernel (device side)
    make_naive_tensor_view →
        make_naive_tensor_descriptor(shape, strides, vec_len, vec_stride)
        + make_buffer_view<global>(p_a, element_space_size)
        = tensor_view{buffer_view, descriptor}

Step 3: Using the tensor view in GridGemm
    make_tile_window(a_grid, (kMPerBlock, kKPerBlock), {iM, 0})
    → a_block_window covering rows [iM, iM+kMPerBlock), K columns [0, kKPerBlock)

Step 4: In the block pipeline
    load_tile(a_copy_dram_window)  → vectorized loads, each thread loads K1=8 fp16 per call
    store_tile(a_copy_lds_window, a_block_tile) → writes to LDS
    a_lds_gemm_window → feeds MFMA
```

---

## Summary

The `Gemm` entry point transforms raw GPU memory into structured, multi-dimensional tensors
through a three-layer abstraction:

1. **Raw Memory**: Linear array of bytes in GPU DRAM
2. **Buffer View**: Adds size, address space, and vectorized access methods
3. **Tensor View**: Adds shape, strides, and multi-dimensional indexing

This abstraction enables:
- Clean, readable code with compile-time shape information
- Type-safe multi-dimensional access
- Automatic vectorization via `number<kAlignment>{}` guarantees
- Flexible memory space handling (global → LDS → registers)
- Efficient tile-based computation via `make_tile_window`

The tensor views created here are passed to `GridGemm`, which orchestrates the block-level
GEMM computation through the three-level hierarchy:
`GridGemm → BlockGemmPipelineAGmemBGmemCReg → BlockGemmASmemBSmemCReg (MFMA)`.
