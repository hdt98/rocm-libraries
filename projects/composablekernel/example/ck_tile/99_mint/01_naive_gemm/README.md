# MINT Naive GEMM Example

This example demonstrates MINT's tile programming model for GEMM (General Matrix Multiplication), based on the structure of `tutorial/ck_tile/gemm/01_naive_gemm/`.

## Status

**Working**: A simplified MINT GEMM implementation (`gemm_simple_mint.hpp`) that successfully builds, runs, and passes verification.

**Educational**: The hierarchical structure (warp/block/host levels) demonstrates the intended design pattern but currently has compilation issues due to MINT API limitations with constexpr in device code.

## Overview

Computes `C[M, N] = A[M, K] × B[N, K]` using MINT's three-level hierarchy:
- **Grid level**: Distributes work across thread blocks
- **Block level**: Manages data movement (global ↔ shared memory) and coordinates warps
- **Warp level**: Performs actual matrix multiplication using warp-level primitives

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Grid Level (host_level/grid_gemm.hpp)                      │
│ - Divides output matrix into block tiles                   │
│ - Each block processes one tile independently               │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Block Level (block_level/)                                  │
│ - BlockGemmPipelineAGmemBGmemCReg: Main pipeline           │
│ - Loads A, B tiles from global → shared memory            │
│ - Coordinates multiple warps to process block tile         │
│ - Policy: Defines tile distributions                       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Warp Level (warp_level/)                                    │
│ - BlockGemmASmemBSmemCReg: Warp-level GEMM                 │
│ - Loads A, B tiles from shared memory                      │
│ - Performs matrix multiplication using SIMT operations     │
│ - Accumulates results in VGPRs (registers)                 │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
01_naive_gemm/
├── gemm_simple_mint.hpp           # WORKING: Simple MINT GEMM implementation
├── practice_gemm.hpp              # Main kernel (uses gemm_simple_mint.hpp)
├── practice_gemm.cpp              # Host code and entry point
├── host_level/                    # EDUCATIONAL: Hierarchical structure
│   └── grid_gemm.hpp              # Grid-level orchestration (has build issues)
├── block_level/                   # EDUCATIONAL: Block-level components
│   ├── block_gemm_pipeline_agmem_bgmem_creg.hpp        # Block pipeline (has build issues)
│   └── block_gemm_pipeline_agmem_bgmem_creg_policy.hpp # Block-level policy
├── warp_level/                    # EDUCATIONAL: Warp-level components
│   ├── block_gemm_asmem_bsmem_creg.hpp                 # Warp GEMM (has build issues)
│   └── block_gemm_asmem_bsmem_creg_policy.hpp          # Warp-level policy
├── CMakeLists.txt
└── README.md
```

The hierarchical files (host_level/, block_level/, warp_level/) show the intended structure matching CK_Tile's `01_naive_gemm` tutorial, but encounter MINT API limitations when called from device code (`make_simple_distribution` uses STL internally, which isn't constexpr-compatible).

## Key MINT Concepts Demonstrated

### 1. Distributed Tensors
```cpp
auto c_acc_dist_tensor = distributed_tensor<
    c_warp_dstr,
    c_warp_elem_layout,
    owned_vgpr_memory<AccDataType, ...>>{};
```
- Distributes data across threads/warps
- Defines partition and element dimensions
- Backed by VGPRs, shared memory, or global memory

### 2. Tensor Descriptors and Views
```cpp
const auto a_gmem_desc = make_aliased_naive_tensor_descriptor(
    sequence<alias_t, alias_t{"M"}, alias_t{"K"}>{},
    alias_t{"Offset"},
    nd_index<2>{M, K},
    nd_index<2>{K, 1});

const auto a_gmem_view = make_tensor_view(
    a_gmem_desc,
    make_global_memory_view(p_a, ...));
```
- Logical tensor layout abstraction
- Separates shape/strides from memory location
- Enables flexible memory access patterns

### 3. Distributed Windows
```cpp
auto a_block_gmem_window = make_distributed_window(
    a_gmem_view,
    nd_index<2>{m_block_start, 0},
    constant<a_block_copy_dstr>{},
    constant<a_block_copy_elem_layout>{},
    constant<ThreadBlock{}>{});
```
- Sliding windows over tensors
- Distributed across cooperating threads
- Supports `move_window()` for iteration

### 4. Tile Operations
```cpp
const auto a_block_tile = load(a_block_gmem_window, gmem_mask);
store(a_block_smem_window, a_block_tile, smem_mask);
```
- Cooperative load/store across threads
- Automatic data distribution
- Masking for boundary conditions

### 5. Warp-Level Matrix Multiplication
```cpp
tile::simt::warp::matmul_mn_mk_kn_no_shuffle(
    c_dist_tensor, a_warp_tile, b_warp_tile);
```
- SIMT-based matmul (portable across backends)
- Can be replaced with XDL/MFMA for ROCm
- Operates on distributed tensors

## Configuration

### Tile Sizes
- **Block tile**: 128×128 (M×N), 16 (K)
- **Warp tile**: 32×32 (M×N), 8 (K)
- **Warp config**: 4 warps in M, 1 warp in N

### Thread Organization
- **Block size**: 256 threads (4 warps × 64 threads/warp)
- Each warp processes 32×32 output elements
- 4 warps tile vertically (M dimension)

## Building

```bash
cd build
ninja example_mint_naive_gemm
```

## Running

```bash
# Default size (2048×2048×2048)
./bin/example_mint_naive_gemm

# With verification
./bin/example_mint_naive_gemm 1

# Custom size with verification
./bin/example_mint_naive_gemm 1 1024 1024 512
```

## Performance Characteristics

Current implementation: ~1.02 TFlops on gfx942 (2048x2048x2048, FP16)

**Working Simple Implementation** (`gemm_simple_mint.hpp`):
- Basic blocked GEMM with shared memory tiling
- Demonstrates MINT's basic concepts without complex hierarchical abstractions
- Educational baseline for learning MINT programming

**Limitations (compared to optimized implementations):**
- Simple thread-to-data mapping
- No tile prefetching or double buffering
- Basic shared memory layout (no swizzling)
- Uses scalar operations instead of MFMA/XDL
- Hierarchical structure shown in separate files has API limitations

**Next Steps for Full Hierarchical Version:**
The hierarchical structure files show the intended design but need:
1. MINT API improvements for constexpr compatibility in device code
2. Manual polymorpher construction instead of `make_simple_distribution`
3. Following patterns from `mint/test/kernel/simt/test_kernel_simt_gemm.cu` (manually construct distributions)

## Comparison to CK_Tile Tutorial

| Aspect | CK_Tile | MINT |
|--------|---------|------|
| **Tensor views** | `make_naive_tensor_view` | `make_tensor_view` + descriptor |
| **Windows** | `make_tile_window` | `make_distributed_window` |
| **Distributions** | `tile_distribution_encoding` | `make_simple_distribution` |
| **Load/Store** | `load_tile` / `store_tile` | `load` / `store` |
| **Warp matmul** | `WarpGemm` (MFMA) | `matmul_mn_mk_kn_no_shuffle` |
| **Sync** | `block_sync_lds()` | `__syncthreads()` |

## Next Steps

To achieve production-level performance:

1. **Use MFMA instructions**: Replace SIMT matmul with `mint::tile::rocm::matmul_xdl`
2. **Add prefetching**: Double-buffer global→shared loads
3. **Optimize shared memory**: Add padding/swizzling to avoid bank conflicts
4. **Tune tile sizes**: Experiment with different M/N/K block sizes
5. **Add async copies**: Use `load_vectorized_freezed_dims` with async copy
6. **Unroll loops**: Use `static_for_n` for compile-time unrolling

See `mint/test/kernel/simt/test_kernel_simt_gemm_optimize.cu` for advanced optimizations.

## Related Files

- `example/ck_tile/99_mint/00_copy_kernel/`: Simpler MINT copy kernel example
- `mint/test/kernel/simt/test_kernel_simt_gemm.cu`: MINT GEMM test with SIMT
- `tutorial/ck_tile/gemm/01_naive_gemm/`: Original CK_Tile tutorial

## Learning Path

1. Start with `00_copy_kernel` to understand basic MINT primitives
2. Study this example to learn hierarchical structure
3. Read MINT headers in `include/mint/` for API details
4. Examine `mint/test/` for advanced usage patterns
5. Implement optimizations incrementally

## References

- MINT documentation: `include/mint/tensor/*.md`
- MINT tile operations: `include/mint/tile/`
- Test examples: `mint/test/kernel/simt/`
