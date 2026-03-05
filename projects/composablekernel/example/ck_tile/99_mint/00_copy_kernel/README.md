# MINT Framework: Getting Started with Tile Copy Operations

## Overview

### Copy Kernel using MINT
A minimal MINT memory copy implementation demonstrating the basic setup required to write a kernel using MINT (Metal INterface Tiles). This tutorial shows how to use MINT as a portable tile abstraction layer while leveraging CK_Tile for host utilities.

MINT provides a next-generation portable tile programming model that works across ROCm and CUDA platforms. This example uses MINT for the core tile operations while using CK_Tile for host-side utilities (memory management, kernel launch, validation).

**Supported Data Type**: fp16 only (MINT's current limitation)

## Build

```bash
# In the root of the repository
mkdir build && cd build
# You can replace <arch> with the appropriate architecture
# (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh ../ <arch>
# Make the copy kernel executable
make mint_tutorial_copy -j
```

This will result in an executable `build/bin/test_mint_copy`

## Example Usage

```bash
./bin/test_mint_copy -m 64 -n 128 -v 1
```

### Arguments:
- `-m`: Input matrix rows (default 64)
- `-n`: Input matrix cols (default 128)
- `-v`: Validation flag to check device results (default 1)
- `-warmup`: Number of warmup iterations (default 50)
- `-repeat`: Number of iterations for kernel execution time (default 100)

## MINT vs CK_Tile Architecture

### Key Differences

| Aspect | CK_Tile | MINT |
|--------|---------|------|
| **Purpose** | AMD-specific optimized tile framework | Portable tile abstraction layer |
| **Platform** | ROCm/HIP only | ROCm and CUDA |
| **Abstraction** | Shape→Problem→Policy→Kernel | Tensor Descriptors + Distributions |
| **Memory Ops** | `load_tile()`, `store_tile()` | `warp::masked_load()`, `warp::masked_store()` |
| **Distribution** | `tile_distribution_encoding` | `make_simple_distribution()` |

### MINT Components

The MINT framework uses these key components:

#### **1. Tensor Descriptors**
Define the logical and physical structure of tensors:

```cpp
auto tensor_desc = make_aliased_naive_packed_tensor_descriptor(
    dim_aliases<"M", "N">{},      // dimension names
    dim_alias<"Offset">{},         // offset alias
    {M, N});                       // tensor dimensions
```

**Purpose**: Describes tensor layout, dimensions, and memory organization.

#### **2. Tensor Views**
Combine memory buffers with tensor descriptors:

```cpp
auto tensor_view = make_tensor_view(
    tensor_desc,
    make_global_memory_view(p_data, tensor_desc.bottom_lengths()[0]));
```

**Purpose**: Provides a view into actual memory with structure defined by the descriptor.

#### **3. Distributions**
Define how data is distributed across threads/warps:

```cpp
constexpr auto dstr = make_simple_distribution(
    index_sequence<RowPerTile, ColPerTile>{},  // tile dimensions
    thread_in_this_warp{},                      // partition strategy
    dim_aliases<"M", "N">{},                   // top-level dims
    dim_aliases<"Lane">{});                     // partition dims
```

**Purpose**: Specifies work distribution pattern across hardware threads.

#### **4. Element Layout**
Defines how elements are organized within each thread's portion:

```cpp
constexpr auto element_layout = make_aliased_naive_packed_tensor_descriptor(
    make_index_sequence<dstr.element_ndim()>{},
    index_constant<-1>{},
    dstr.element_lengths());
```

**Purpose**: Describes per-thread element organization.

#### **5. Distributed Windows**
Windows into tensors with thread distribution:

```cpp
auto window = make_distributed_window(
    tensor_view,             // tensor to window into
    {row_origin, col_origin}, // window origin
    constant<dstr>{},        // distribution
    constant<element_layout>{}, // element layout
    constant<thread_in_this_warp{}>{});  // partition
```

**Purpose**: Provides a distributed view of a tile, managing thread-level access.

### Hybrid Approach in This Tutorial

This tutorial demonstrates using **MINT for core operations** while **CK_Tile for host utilities**:

- **MINT Used For**:
  - Tensor descriptors and views
  - Tile distributions
  - Load/store operations (`warp::masked_load`, `warp::masked_store`)
  - Window management

- **CK_Tile Used For**:
  - Host memory management (`ck_tile::DeviceMem`, `ck_tile::HostTensor`)
  - Kernel launch infrastructure (`ck_tile::launch_kernel`)
  - Validation utilities (`ck_tile::check_err`)
  - Argument parsing (`ck_tile::ArgParser`)

## Core MINT Concepts

### Aliased Tensor Descriptors

MINT uses named dimensions (aliases) instead of numeric indices:

```cpp
// Instead of dimensions [0] and [1], use "M" and "N"
dim_aliases<"M", "N">{}
```

**Benefits**:
- Self-documenting code
- Safer dimension manipulation
- Clearer intent

### Simple Distribution

The simple distribution pattern distributes data across a warp:

```cpp
constexpr auto dstr = make_simple_distribution(
    index_sequence<16, 16>{},       // 16x16 tile per warp
    thread_in_this_warp{},          // distribute across warp threads
    dim_aliases<"M", "N">{},        // working on M and N dimensions
    dim_aliases<"Lane">{});         // partition by lane ID
```

This creates a mapping where each thread (lane) in the warp handles a portion of the 16x16 tile.

### Masked Load/Store

MINT uses masked operations for boundary handling:

```cpp
// Load with optional masking
const auto tile = warp::masked_load(src_window, src_mask);

// Store with optional masking
warp::masked_store(dst_window, dst_mask, tile);
```

The mask parameter (can be empty `tuple<>{}`) allows selective load/store for boundary conditions.

### Window Movement

Moving windows to process multiple tiles:

```cpp
move_window(window, {delta_m, delta_n});
```

This updates the window origin to point to the next tile.

## The Kernel Structure

### Kernel Implementation

The `MintCopyKernel` implements a basic warp-level copy operation:

```cpp
template <index_t RowPerTile, index_t ColPerTile, typename T>
__global__ void MintCopyKernel(T* p_dst, const T* p_src,
                                index_t M, index_t N)
{
    // 1. Create tensor descriptors
    // 2. Create tensor views
    // 3. Create distributions and element layouts
    // 4. Create distributed windows
    // 5. Loop over tiles and copy data
}
```

### Step-by-Step Execution

1. **Tensor Descriptor Creation**:
   ```cpp
   const auto src_desc_m_n = make_aliased_naive_packed_tensor_descriptor(
       dim_aliases<"M", "N">{}, dim_alias<"SrcOffset">{}, {M, N});
   ```
   Defines the logical structure of source tensor.

2. **Tensor View Creation**:
   ```cpp
   const auto src_view = make_tensor_view(
       src_desc_m_n,
       make_global_memory_view(p_src, src_desc_m_n.bottom_lengths()[0]));
   ```
   Combines descriptor with actual memory buffer.

3. **Distribution Setup**:
   ```cpp
   constexpr auto dstr = make_simple_distribution(
       index_sequence<RowPerTile, ColPerTile>{},
       thread_in_this_warp{},
       dim_aliases<"M", "N">{},
       dim_aliases<"Lane">{});
   ```
   Defines how the tile is distributed across warp threads.

4. **Distributed Window Creation**:
   ```cpp
   auto src_window = make_distributed_window(
       src_view, {0, 0}, constant<dstr>{},
       constant<element_layout>{}, constant<thread_in_this_warp{}>{});
   ```
   Creates a warp-distributed view into the source tensor.

5. **Tile Processing Loop**:
   ```cpp
   for (index_t col = 0; col < N; col += ColPerTile) {
       const auto tile = warp::masked_load(src_window, src_mask);
       warp::masked_store(dst_window, dst_mask, tile);
       move_window(src_window, {0, ColPerTile});
       move_window(dst_window, {0, ColPerTile});
   }
   ```

## Memory Access Patterns

### Warp-Level Operations
- Each warp processes a `RowPerTile × ColPerTile` tile
- Work is distributed across 32 (CUDA) or 64 (ROCm) threads
- Enables coalesced memory access

### Distribution Strategy
The simple distribution ensures:
- All threads in a warp participate
- Memory accesses are coalesced
- Efficient use of memory bandwidth

## Portability

MINT kernels can compile for both ROCm and CUDA:

```cpp
#if defined(MINT_BACKEND_ROCM)
    // ROCm-specific code (if needed)
#elif defined(MINT_BACKEND_CUDA)
    // CUDA-specific code (if needed)
#endif
```

The same MINT code works on both platforms, with platform-specific optimizations applied automatically.

## Performance Considerations

1. **Tile Size**: Balance between parallelism and memory usage
2. **Warp Utilization**: Ensure full warp participation
3. **Memory Coalescing**: Distribution pattern affects coalescing
4. **Boundary Handling**: Masked operations handle irregular sizes

## Next Steps

After mastering this basic copy kernel, explore:
- More complex distributions
- LDS (shared memory) staging
- Element-wise operations during copy
- Multiple data types (when MINT support expands beyond fp16)
- Integration with MINT's matmul operations
