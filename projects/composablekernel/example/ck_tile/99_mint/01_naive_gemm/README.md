# MINT Naive GEMM Example

This example demonstrates a basic GEMM (General Matrix Multiplication) implementation using MINT (Metal INterface Tiles), showcasing MINT's tensor view abstraction.

## Overview

Computes `C = A × B` where:
- A is an M×K matrix
- B is an N×K matrix (stored transposed)
- C is an M×N matrix

The implementation uses:
- **MINT tensor views** (`make_global_packed_tensor_view`) for accessing global memory
- **Shared memory** for block-level tile caching
- **Thread-level parallelism** for computation within each block

## Key MINT Concepts Demonstrated

1. **Tensor Views**: Abstraction over memory with logical indexing
   ```cpp
   auto a_view = make_global_packed_tensor_view(p_a, nd_index<2>{M, K});
   ```

2. **Element Access**: Clean indexing through tensor views
   ```cpp
   a_view.element(nd_index<2>{row, col})
   ```

3. **Type Conversions**: Proper handling of mixed precision (FP16 inputs, FP32 accumulation)

## Implementation Structure

- **Cooperative Loading**: All threads in a block cooperatively load tiles into shared memory
- **Computation**: Each thread computes a subset of the output tile
- **Synchronization**: `__syncthreads()` ensures data consistency

## Tile Configuration

- Block tile: 128×128 output elements
- K-dimension tile: 16
- Thread block: 16×16 threads (256 total)
- Each thread computes: 8×8 output elements

## Building

```bash
cd build
ninja example_mint_naive_gemm
```

## Running

```bash
# Run with default size (2048×2048×2048)
./bin/example_mint_naive_gemm

# Run with verification
./bin/example_mint_naive_gemm 1

# Run with custom size and verification
./bin/example_mint_naive_gemm 1 1024 1024 512
```

## Performance Characteristics

This is a **naive implementation** designed for clarity, not peak performance:
- Uses simple shared memory without swizzling
- No prefetching or double buffering
- No MFMA/XDL instructions (uses scalar operations)
- Straightforward thread-to-data mapping

For production GEMM kernels, see the optimized CK_Tile examples that use:
- Warp-level MFMA instructions
- Complex tile distributions
- Prefetching and pipelining
- Bank conflict avoidance

## Next Steps

To understand more advanced MINT concepts:
1. Study distributed tensors for warp-level cooperation
2. Learn about polymorphers for flexible memory layouts
3. Explore MFMA integration with `mint::tile::rocm::matmul_xdl`
4. Investigate tilers for systematic tile iteration
