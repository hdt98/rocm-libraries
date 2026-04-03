# Naive GEMM: Step-by-Step Code Walkthrough

This document provides a detailed walkthrough of `practice_gemm.cpp`, explaining each step of
implementing a GEMM (General Matrix Multiplication) kernel using the CK Tile API.

## Overview

We implement `C = A × B` where:
- `A` is an `M × K` matrix
- `B` is an `N × K` matrix (B is stored in `[N, K]` layout — N as the leading dimension, K contiguous)
- `C` is an `M × N` matrix

The implementation uses a hierarchical tiling strategy with two levels:
1. **Block Tiles**: Processed by one thread block (256 threads)
2. **Warp Tiles**: Processed by warps (wavefronts) within blocks using MFMA instructions

---

## Step 1: Define Data Types

```cpp
using ADataType   = ck_tile::half_t;   // FP16 input
using BDataType   = ck_tile::half_t;   // FP16 input
using AccDataType = float;              // FP32 accumulation (numerical precision)
using CDataType   = ck_tile::half_t;   // FP16 output (cast from FP32 accumulator)
```

**What's happening:**
- We use `half_t` (FP16) for input matrices A and B — efficient memory bandwidth and high MFMA throughput.
- The accumulator uses `float` (FP32) to prevent precision loss from repeated FP16 additions.
- The output matrix C uses `half_t`: the FP32 accumulator is cast to FP16 via `type_convert<CDataType>` in `grid_gemm.hpp`.

Note: Unlike simplified examples that use `float` for C, the production kernel uses `half_t` for C to match typical ML workload requirements.

---

## Step 2: Define Problem Size

```cpp
ck_tile::index_t M = 3328;   // Number of rows in A and C (default)
ck_tile::index_t N = 4096;   // Number of columns in B and C (default)
ck_tile::index_t K = 4096;   // Inner dimension: cols of A, rows of B (default)
```

These defaults are representative of large language model (LLM) shapes. The binary accepts optional
command-line arguments to override them:

```bash
./tile_tutorial_naive_gemm                      # use defaults M=3328, N=4096, K=4096
./tile_tutorial_naive_gemm 1                    # enable verification, keep defaults
./tile_tutorial_naive_gemm 1 512 256 64         # enable verification, custom M N K
```

**Memory layout:**
```
Matrix A [M, K]:        Matrix B [N, K]:        Matrix C [M, N]:
stride_a = K            stride_b = K            stride_c = N
(row-major)             (N is leading dim)       (row-major)
```

B is intentionally stored in `[N, K]` (B-transposed) layout. Each row of B in memory is a
full K-vector, so loading "rows" of B produces the K-aligned vectors that MFMA expects —
enabling both coalesced and vectorized global loads without any in-kernel transposition.

---

## Step 3: Create Host Tensors

```cpp
auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};   // N×K, not K×N
auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};

auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};  // row-major
auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};  // N-major (B transposed)
auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};  // row-major

ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);  // will receive GPU results
```

`HostTensor` is a CK Tile utility that manages CPU memory. Each tensor is defined by its shape
(`lengths`) and memory strides (`strides`). Stride `1` for the last dimension means elements
in that dimension are contiguous — the precondition for vectorized loads.

---

## Step 4: Initialize Tensors with Random Data

```cpp
ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);
```

`FillUniformDistributionIntegerValue` fills with random **integer-valued** data in `[-5, 5]`
(stored as FP16). Using integer values avoids FP16 rounding noise during verification —
the CPU reference computation in FP32 and the GPU computation with FP32 accumulation will
agree exactly for integer-valued inputs within this range.

---

## Step 5: Allocate Device Memory and Transfer Data

```cpp
// Two-step pattern: explicit size allocation + explicit transfer
ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
a_buf.ToDevice(a_host.mData.data());

ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
b_buf.ToDevice(b_host.mData.data());

ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());
```

The two-step pattern (allocate then transfer) is explicit: `DeviceMem(size)` calls `hipMalloc`,
and `ToDevice(ptr)` calls `hipMemcpy`. This is in contrast to one-shot constructors; the
explicit pattern makes the allocation and transfer steps visible and independently verifiable.

Note: `c_buf` is allocated but **not** initialized before the kernel launch — the kernel
writes all of C from scratch.

**Memory flow:**
```
CPU (Host)             GPU (Device)
┌─────────┐           ┌─────────┐
│ a_host  │ ToDevice> │  a_buf  │
│ b_host  │ ToDevice> │  b_buf  │
│         │           │  c_buf  │  ← kernel writes here
└─────────┘           └─────────┘
```

---

## Step 6: Configure Block Tile Shape

```cpp
// Block tile: the region of C each thread block computes in a single kernel launch
constexpr ck_tile::index_t kGemmMPerBlock = 256;  // M dimension per block
constexpr ck_tile::index_t kGemmNPerBlock = 128;  // N dimension per block
constexpr ck_tile::index_t kGemmKPerBlock = 32;   // K slice per loop iteration

// Alignment = 8 means 8 fp16 elements per load = 128 bits = one global_load_dwordx4
constexpr ck_tile::index_t kAAlignment = 8;
constexpr ck_tile::index_t kBAlignment = 8;
constexpr ck_tile::index_t kCAlignment = 8;

// 2 blocks per CU: 4 SIMDs/CU × 2 warps/SIMD → 8 warps/CU, then /4 warps/block = 2 blocks/CU
constexpr ck_tile::index_t kWarpPerCu   = 8;
constexpr ck_tile::index_t kBlockPerCu  = kWarpPerCu / (kBlockSize / warp_size);  // = 2
```

Each thread block (256 threads = 4 warps) computes a 256×128 tile of C, iterating over K
in slices of 32 elements. The total number of blocks launched equals the number of tiles
needed to cover C:

```
kGridSize = ceil(M / kGemmMPerBlock) * ceil(N / kGemmNPerBlock)
          = ceil(3328 / 256) * ceil(4096 / 128)
          = 13 * 32
          = 416 blocks
```

### About the MFMA instruction used

The actual hardware MFMA instruction is **v_mfma_f32_32x32x8f16**:
- Output: 32×32 matrix (kM=32, kN=32)
- K reduction width: 8 fp16 elements per call (kK=8)
- 64 threads per warp, each owning a slice of the 32×32 output

The block tile (256×128×32) is divided among 4 warps (MWarp=4, NWarp=1):
- `MIterPerWarp = 256 / (4 * 32) = 2` — each warp covers 2 M-tiles of size 32
- `NIterPerWarp = 128 / (1 * 32) = 4` — each warp covers 4 N-tiles of size 32
- `KIterPerWarp = 32  / 8        = 4` — each K-slice needs 4 MFMA calls

Total MFMA calls per warp: 2 × 4 × 4 = 32. Total per block: 128.

---

## Step 7: Construct the Kernel Type

```cpp
// Gemm struct bundles all configuration at compile time.
// All template parameters are resolved here; the kernel itself has no runtime decisions.
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

`Gemm` is the top-level functor defined in `practice_gemm.hpp`. It wraps a `GridGemm` (host-level
dispatcher) and a `GridGemmPolicy` (policy struct) plus a `BlockGemmPipelineAGmemBGmemCReg`
(block pipeline). All these types are resolved at compile time by the single `Gemm` instantiation.

**CK Tile design pattern:**
```
Kernel = Problem + Policy + Epilogue
         ↑         ↑        ↑
      (What)    (How)   (Post-processing)
```

---

## Step 8: Calculate Grid/Block Dimensions

```cpp
ck_tile::index_t kGridSize = ck_tile::integer_divide_ceil(M, kGemmMPerBlock) *
                              ck_tile::integer_divide_ceil(N, kGemmNPerBlock);

constexpr ck_tile::index_t kBlockSize = 256;  // 4 warps of 64 threads each
```

- `kGridSize`: Total number of thread blocks = total C tiles to compute
- `kBlockSize`: Fixed at 256 threads — 4 warps × 64 threads/warp (AMD CDNA)
- `kBlockPerCu = 2`: Occupancy hint. With 4 SIMDs/CU and 2 warps/SIMD (8 warps/CU total)
  and 4 warps/block, this places 2 blocks per CU simultaneously.

---

## Step 9: Launch the Kernel

```cpp
float ave_time = ck_tile::launch_kernel(
    ck_tile::stream_config{nullptr, true, 0, 5, 1000},
    //                     ↑       ↑     ↑  ↑  ↑
    //                     stream  time  log warmup repeat
    ck_tile::make_kernel<kBlockPerCu>(gemm_kernel{},
        kGridSize, kBlockSize, 0,
        a_buf, b_buf, c_buf, M, N, K, Lda, Ldb, Ldc, CElementFunction{}));
```

**`stream_config` parameters:**
- `nullptr`: Use the default HIP stream
- `true`: Enable timing (measure kernel execution time)
- `0`: Log level (0 = minimal output)
- `5`: Warm-up iterations (run kernel 5 times before timing)
- `1000`: Timed iterations (average over 1000 runs for stable measurement)

**`make_kernel<kBlockPerCu>`**: Wraps the kernel functor with the occupancy hint.
`kBlockPerCu=2` tells the runtime to schedule 2 blocks per CU at a time.

**Kernel arguments**: Raw device buffer pointers (not typed) and the problem dimensions
(`M, N, K, Lda, Ldb, Ldc`) plus the epilogue functor `CElementFunction{}`.

### Kernel execution flow

```
launch_kernel() → gemm_kernel::operator()()
    ↓
Gemm::operator() calls GridGemm<Problem, Policy>::operator()
    ↓
map block_id → 2D tile (iM, iN)
create a_block_window, b_block_window
allocate __shared__ p_smem
    ↓
BlockGemmPipelineAGmemBGmemCReg::operator()
    ↓
for each K slice:
    load A,B from DRAM → VGPRs
    store A,B from VGPRs → LDS
    block_sync_lds()
    BlockGemmASmemBSmemCReg (MFMA)
    block_sync_lds()
    ↓
tile_elementwise_in: type_convert(acc) → CDataType, apply CElementFunction
store_tile: VGPRs → DRAM (C matrix)
```

---

## Step 10: Verify Results (Optional)

```cpp
if(verification)
{
    // CPU reference (ground truth)
    ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
    reference_basic_gemm<ADataType, ADataType, AccDataType, CDataType>(
        a_host, b_host, c_host_ref);
    //                  ↑ note: second arg is ADataType not BDataType (both are half_t here)

    // Copy GPU results back to host
    c_buf.FromDevice(c_host_dev.mData.data());

    // Compare element-wise
    bool pass = ck_tile::check_err(c_host_dev, c_host_ref);
    std::cout << "valid: " << (pass ? "y" : "n") << std::endl;
}
```

`reference_basic_gemm` performs the same GEMM on CPU using FP32 accumulation. The GPU
results (cast back from FP16 to be compared) should match the reference within FP16
rounding tolerance. Using integer-valued inputs (Step 4) ensures exact agreement.

**Verification flow:**
```
CPU                      GPU
┌─────────┐             ┌─────────┐
│ a_host  │ ToDevice >  │  a_buf  │
│ b_host  │ ToDevice >  │  b_buf  │
└─────────┘             └─────────┘
     │                       │
     ↓                       ↓
reference_gemm()        kernel launch
     │                       │
     ↓                       ↓
┌──────────┐           ┌──────────┐
│c_host_ref│           │  c_buf   │
└──────────┘           └──────────┘
     │                       │ FromDevice()
     │                       ↓
     │                 ┌──────────┐
     └──── check_err ──│c_host_dev│
                       └──────────┘
```

---

## Complete Execution Flow Summary

```
1. Define data types (ADataType=FP16, BDataType=FP16, AccDataType=FP32, CDataType=FP16)
2. Set problem size (M=3328, N=4096, K=4096 by default)
3. Create host tensors and initialize with random integer-valued data
4. Allocate device memory and transfer data (CPU → GPU) via two-step DeviceMem pattern
5. Configure block tile (kGemmMPerBlock=256, kGemmNPerBlock=128, kGemmKPerBlock=32)
6. Construct gemm_kernel type (fully specialized at compile time via Gemm<...>)
7. Launch kernel (416 blocks for default sizes, 256 threads/block, 2 blocks/CU)
8. Execute GEMM on GPU (DRAM→LDS→MFMA per K slice, accumulate in FP32 VGPRs)
9. Cast FP32 accumulator to FP16, apply CElementFunction, store C to DRAM
10. (Optional) Verify results: compare GPU vs CPU reference
11. Print performance metrics (TFlops, GB/s)
```
