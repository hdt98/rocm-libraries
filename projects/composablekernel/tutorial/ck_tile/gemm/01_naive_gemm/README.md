# CK Tile Practice GEMM Example

This is a practice implementation of a GEMM (General Matrix Multiplication) kernel using the CK Tile API. It demonstrates the fundamental concepts of GPU kernel development using CK Tile's hierarchical tile system.

## CK Tile API Structure

In the composable_kernel library's ck_tile API, **A Kernel is composed of a Problem, a Policy and an Epilogue**:

1. **Problem** describes the shape, data type, data layout, precision of our GEMM matrices
2. **Policy** describes how the data in the matrix (or tile) is mapped to the threads
3. **Epilogue** describes additional computation work performed after the gemm computations (this example uses an identity `CElementFunction` epilogue that passes through values unchanged)

## Overview

This example implements a complete GEMM kernel `C = A × B` using the CK Tile framework, showcasing:

- **Problem Setup** - Setting up the problem (input/output shapes, data types, mathematical operations), composing a kernel (pipeline, policy, epilogue), kernel launch
- **Block-level Pipelining** - creating tensor views, dispatching to block-level GEMM
- **Block-level GEMM Computation** - Block tiles, tile window creation, loading/storing to DRAM and Register memory
- **Warp-level GEMM Computation** - Warp tiles, MFMA level computation

## Data Types

```cpp
using ADataType   = ck_tile::half_t;   // FP16 input
using BDataType   = ck_tile::half_t;   // FP16 input
using AccDataType = float;              // FP32 accumulation (numerical precision)
using CDataType   = ck_tile::half_t;   // FP16 output (cast from AccDataType)
```

Note: C is FP16, not FP32. The accumulation happens in FP32, then each element is cast to FP16 via `type_convert<CDataType>` in `grid_gemm.hpp`.

## Problem Setup and Data Flow

### Problem Size Configuration
We set the problem size using the M, N and K variables:
```cpp
ck_tile::index_t M = 3328;   // Number of rows in A and C
ck_tile::index_t N = 4096;   // Number of columns in B and C
ck_tile::index_t K = 4096;   // Number of columns in A, rows in B
```

### Host Matrix Creation
Three host matrices A (M×K), B (N×K) and C (M×N) are created, initialized on the CPU and copied over to the GPU global/DRAM memory.

**B is stored in [N, K] layout** (N as leading dimension, K contiguous). This is the transposed form: each N row contains a full K-vector. This enables vectorized and coalesced global loads from B without any transposition in the kernel.

```cpp
// Host tensors with proper strides
ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);  // M × K
ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);  // N × K  (B transposed)
ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);  // M × N

// Initialize with random integer-valued data in [-5, 5]
ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

// Allocate device memory (explicit two-step pattern)
ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
a_buf.ToDevice(a_host.mData.data());
```

### Block Tile Configuration
A block tile defines the region of C each thread block computes:

```cpp
constexpr ck_tile::index_t kGemmMPerBlock = 256;  // M dimension per block
constexpr ck_tile::index_t kGemmNPerBlock = 128;  // N dimension per block
constexpr ck_tile::index_t kGemmKPerBlock = 32;   // K dimension per K-loop iteration
```

- A BlockTile of size MxK (256x32) on A matrix and NxK (128x32) on B matrix.
- BlockTiles iterate in K dimension to fetch data required for computing region of C covered by C's block tile.

### About the MFMA Instruction (WarpGemm)
The actual MFMA hardware instruction used is **32×32×8 fp16**:
- `WarpGemm::kM = 32`: 32 output rows per MFMA call
- `WarpGemm::kN = 32`: 32 output columns per MFMA call
- `WarpGemm::kK = 8`: 8 K elements consumed per MFMA call

The block tile (256×128×32) is divided among 4 warps (MWarp=4, NWarp=1), each computing a sub-region:
- `MIterPerWarp = 256 / (4 * 32) = 2` (each warp covers 2 M-tiles)
- `NIterPerWarp = 128 / (1 * 32) = 4` (each warp covers 4 N-tiles)
- `KIterPerWarp = 32  / 8        = 4` (each K-slice needs 4 MFMA calls)

Total MFMA calls per warp: 2 × 4 × 4 = 32. Total per block: 128.

The MFMA variant is selected by `CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION` (default: 1, transposed layout).

### Problem and Policy Composition
```cpp
// Gemm struct bundles all configuration at compile time
using gemm_kernel = ck_tile::Gemm<ADataType,
                                   BDataType,
                                   AccDataType,
                                   CDataType,
                                   CElementFunction,
                                   kAAlignment,
                                   kBAlignment,
                                   kCAlignment,
                                   kBlockSize,
                                   kGemmMPerBlock,
                                   kGemmNPerBlock,
                                   kGemmKPerBlock>;
```

### Kernel Launch
`ck_tile::launch_kernel()` is used to launch the kernel on device:
```cpp
float ave_time = ck_tile::launch_kernel(
    ck_tile::stream_config{nullptr, true, 0, 5, 1000},  // stream, timing, log, warmup, repeat
    ck_tile::make_kernel<kBlockPerCu>(gemm_kernel{},
        kGridSize, kBlockSize, 0,
        // Kernel arguments: device buffers and problem dimensions
        a_buf, b_buf, c_buf, M, N, K, Lda, Ldb, Ldc, CElementFunction{}));
```

### Result Verification
The results from the kernel are compared with results from CPU based computation function:
```cpp
// CPU reference implementation
ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
reference_basic_gemm<ADataType, ADataType, AccDataType, CDataType>(a_host, b_host, c_host_ref);

// Copy GPU results back to host
c_buf.FromDevice(c_host_dev.mData.data());

// Verify correctness
bool pass = ck_tile::check_err(c_host_dev, c_host_ref);
```

### Runtime Flow

The main program (`practice_gemm.cpp`) is the entry point for the runtime flow:

```
1. Define data types (ADataType=FP16, BDataType=FP16, AccDataType=FP32, CDataType=FP16)
2. Set problem size (M=3328, N=4096, K=4096)
3. Create host tensors and initialize with random integer-valued data
4. Allocate device memory and transfer data (CPU → GPU)
5. Configure block tile shapes (kGemmMPerBlock=256, kGemmNPerBlock=128, kGemmKPerBlock=32)
6. Construct gemm_kernel type (fully specialized at compile time)
7. Launch kernel (416 blocks for default sizes, 256 threads/block, 2 blocks/CU)
8. Execute GEMM on GPU (DRAM→LDS→MFMA per K slice, accumulate in FP32 VGPRs)
9. Cast FP32 accumulator to FP16, apply CElementFunction, store C to DRAM
10. (Optional) Verify results: compare GPU vs CPU reference
11. Print performance metrics (TFlops, GB/s)
```

## Building and Running

```bash
# From composable_kernel root directory
mkdir build && cd build
../script/cmake-ck-dev.sh ../ <arch>
make tile_tutorial_naive_gemm -j

# Run with sample sizes
./bin/tile_tutorial_naive_gemm
```

This example serves as a foundation for understanding more complex GEMM implementations and optimization strategies in the CK Tile framework.
