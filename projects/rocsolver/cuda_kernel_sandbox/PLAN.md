# rocSOLVER CUDA Kernel Sandbox - Implementation Plan

## Overview

This plan describes a minimal "kernel sandbox suite" that allows compiling individual rocSOLVER kernels for CUDA using nvcc. The purpose is to enable testing rocSOLVER kernels with CUDA Compute Sanitizer tools (memcheck, racecheck, initcheck, synccheck).

**Key Constraint**: rocSOLVER cannot be compiled in its entirety for CUDA due to rocBLAS dependencies and HIP runtime incompatibilities. This sandbox isolates individual kernels with minimal stubs.

## File Structure

```
cuda_kernel_sandbox/
+-- cuda_compat.cuh        # HIP-to-CUDA compatibility layer
+-- rocsolver_types.cuh    # Minimal type definitions
+-- device_helpers.cuh     # Device helper functions (aabs, swap, load_ptr_batch)
+-- kernels/
|   +-- getf2_small_kernel.cuh   # The kernel under test (modified)
+-- sandbox_getf2_small.cu       # Main driver for getf2_small_kernel
+-- Makefile                     # Build system
+-- PLAN.md                      # This file
```

## Detailed File Contents

### 1. `cuda_compat.cuh` - HIP-to-CUDA Compatibility Layer

Maps HIP thread/block indexing to CUDA equivalents:

```cpp
#pragma once

// Map HIP thread/block indexing macros to CUDA
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z
#define hipBlockIdx_x  blockIdx.x
#define hipBlockIdx_y  blockIdx.y
#define hipBlockIdx_z  blockIdx.z
#define hipBlockDim_x  blockDim.x
#define hipBlockDim_y  blockDim.y
#define hipBlockDim_z  blockDim.z
#define hipGridDim_x   gridDim.x
#define hipGridDim_y   gridDim.y
#define hipGridDim_z   gridDim.z

// Map HIP memory management to CUDA
#define hipMalloc       cudaMalloc
#define hipFree         cudaFree
#define hipMemcpy       cudaMemcpy
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemset       cudaMemset
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetLastError         cudaGetLastError
#define hipSuccess              cudaSuccess
#define hipError_t              cudaError_t
#define hipStream_t             cudaStream_t
```

### 2. `rocsolver_types.cuh` - Minimal Type Definitions

Defines only the types needed by the kernel:

```cpp
#pragma once

#include <cstdint>
#include <cmath>
#include <complex>

// Core rocBLAS types
using rocblas_stride = int64_t;
using rocblas_int = int32_t;

// rocblas_handle is unused in kernel code - just stub it
using rocblas_handle = void*;

// rocblas_status for return values
enum rocblas_status {
    rocblas_status_success = 0,
    // ... other status codes as needed
};

// Complex type support (if needed for complex kernels)
template <typename T>
struct rocblas_complex_num {
    T real_, imag_;
    __device__ __host__ rocblas_complex_num(T r = 0, T i = 0) : real_(r), imag_(i) {}
    __device__ __host__ T real() const { return real_; }
    __device__ __host__ T imag() const { return imag_; }
};

using rocblas_float_complex = rocblas_complex_num<float>;
using rocblas_double_complex = rocblas_complex_num<double>;

// Type trait for complex detection
template <typename T>
constexpr bool rocblas_is_complex = false;

template <>
constexpr bool rocblas_is_complex<rocblas_float_complex> = true;

template <>
constexpr bool rocblas_is_complex<rocblas_double_complex> = true;

// Namespace macros - simplified for sandbox
#define ROCSOLVER_BEGIN_NAMESPACE namespace rocsolver {
#define ROCSOLVER_END_NAMESPACE }

// Kernel macro
#define ROCSOLVER_KERNEL __global__

// Size constants for getf2
#define GETF2_SSKER_MAX_M 512
#define GETF2_SSKER_MAX_N 64
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
```

### 3. `device_helpers.cuh` - Device Helper Functions

Contains device functions used by kernels:

```cpp
#pragma once

#include "rocsolver_types.cuh"

ROCSOLVER_BEGIN_NAMESPACE

// Absolute value helper
template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val);
}

template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val.real()) + std::abs(val.imag()); // asum
}

// Swap helper
template <typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

// Load pointer for batched operations (strided version)
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
load_ptr_batch(T* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p + block * stride + offset;
}

// Load pointer for batched operations (array of pointers version)
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
load_ptr_batch(T* const* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

ROCSOLVER_END_NAMESPACE
```

### 4. `kernels/getf2_small_kernel.cuh` - The Kernel Under Test

Copy of `getf2_small_kernel` from `library/src/specialized/roclapack_getf2_specialized_kernels.hpp` with modifications:

**Modifications from original:**
1. Remove `#include "rocblas.hpp"` and `#include "rocsolver_run_specialized_kernels.hpp"`
2. Add `#include "cuda_compat.cuh"`, `#include "rocsolver_types.cuh"`, `#include "device_helpers.cuh"`
3. **Remove the `__syncthreads()` on line 103** (the bug being investigated)
4. Keep the kernel function itself mostly intact

```cpp
#pragma once

#include "cuda_compat.cuh"
#include "rocsolver_types.cuh"
#include "device_helpers.cuh"

ROCSOLVER_BEGIN_NAMESPACE

template <int DIM, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_SSKER_MAX_M)
    getf2_small_kernel(const I m,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       I* ipivA,
                       const rocblas_stride shiftP,
                       const rocblas_stride strideP,
                       INFO* infoA,
                       const I batch_count,
                       const I offset,
                       I* permut_idx,
                       const rocblas_stride stridePI)
{
    // ... kernel body copied from original ...
    // NOTE: __syncthreads() on line 103 (synchronize across waves
    //       before overwriting common) is REMOVED for testing
}

ROCSOLVER_END_NAMESPACE
```

### 5. `sandbox_getf2_small.cu` - Main Driver

The main function that sets up test data and launches the kernel:

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#include "kernels/getf2_small_kernel.cuh"

using namespace rocsolver;

// Macro to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    // Default parameters (can be overridden via command line)
    int m = 16;           // Number of rows
    int n = 8;            // Number of columns (DIM)
    int batch_count = 1;  // Number of matrices in batch

    // Parse command line args
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) batch_count = atoi(argv[3]);

    printf("Running getf2_small_kernel with m=%d, n=%d, batch=%d\n", m, n, batch_count);

    // Validate parameters
    if (n > 64 || m > 512) {
        fprintf(stderr, "Error: n must be <= 64, m must be <= 512\n");
        return 1;
    }

    // Allocate host memory
    int lda = m;
    size_t matrix_size = (size_t)lda * n;
    size_t total_size = matrix_size * batch_count;

    float* h_A = (float*)malloc(total_size * sizeof(float));
    int* h_ipiv = (int*)malloc(n * batch_count * sizeof(int));
    int* h_info = (int*)calloc(batch_count, sizeof(int));

    // Initialize matrix with random values
    srand(42);
    for (size_t i = 0; i < total_size; i++) {
        h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Allocate device memory
    float* d_A;
    int* d_ipiv;
    int* d_info;

    CUDA_CHECK(cudaMalloc(&d_A, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_count * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_count * sizeof(int)));

    // Calculate grid/block dimensions
    // Based on getf2_run_small launcher logic
    int opval[] = {GETF2_OPTIM_NGRP};
    int ngrp = (batch_count < 2 || m > 32) ? 1 : opval[m - 1];
    int blocks = (batch_count - 1) / ngrp + 1;

    dim3 grid(1, blocks, 1);
    dim3 block(m, ngrp, 1);

    rocblas_stride strideA = matrix_size;
    rocblas_stride strideP = n;

    // Shared memory size
    size_t shmem_size = std::max(m, n) * ngrp * sizeof(float);

    printf("Launching kernel: grid(%d,%d,%d) block(%d,%d,%d) shmem=%zu\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem_size);

    // Launch kernel - need to dispatch based on n (DIM template parameter)
    // For simplicity, just handle a few common cases
    switch(n) {
        case 8:
            getf2_small_kernel<8, float, int, int, float*><<<grid, block, shmem_size>>>(
                m, d_A, 0, lda, strideA, d_ipiv, 0, strideP, d_info, batch_count, 0, nullptr, 0);
            break;
        case 16:
            getf2_small_kernel<16, float, int, int, float*><<<grid, block, shmem_size>>>(
                m, d_A, 0, lda, strideA, d_ipiv, 0, strideP, d_info, batch_count, 0, nullptr, 0);
            break;
        // Add more cases as needed...
        default:
            fprintf(stderr, "Unsupported n=%d. Add case to switch statement.\n", n);
            return 1;
    }

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_A, d_A, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ipiv, d_ipiv, n * batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_info, d_info, batch_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results summary
    printf("Kernel completed.\n");
    printf("info[0] = %d\n", h_info[0]);
    printf("First pivot indices: ");
    for (int i = 0; i < std::min(n, 8); i++) {
        printf("%d ", h_ipiv[i]);
    }
    printf("\n");

    // Cleanup
    free(h_A);
    free(h_ipiv);
    free(h_info);
    cudaFree(d_A);
    cudaFree(d_ipiv);
    cudaFree(d_info);

    return 0;
}
```

### 6. `Makefile`

```makefile
NVCC = nvcc
NVCC_FLAGS = -O2 -g -G -lineinfo
CUDA_ARCH = -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86

# Include paths
INCLUDES = -I. -I./kernels

# Targets
all: sandbox_getf2_small

sandbox_getf2_small: sandbox_getf2_small.cu kernels/getf2_small_kernel.cuh \
                     cuda_compat.cuh rocsolver_types.cuh device_helpers.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o $@ $<

clean:
	rm -f sandbox_getf2_small

# Run with compute-sanitizer
sanitizer-memcheck: sandbox_getf2_small
	compute-sanitizer --tool memcheck ./sandbox_getf2_small

sanitizer-racecheck: sandbox_getf2_small
	compute-sanitizer --tool racecheck ./sandbox_getf2_small

sanitizer-initcheck: sandbox_getf2_small
	compute-sanitizer --tool initcheck ./sandbox_getf2_small

sanitizer-synccheck: sandbox_getf2_small
	compute-sanitizer --tool synccheck ./sandbox_getf2_small

.PHONY: all clean sanitizer-memcheck sanitizer-racecheck sanitizer-initcheck sanitizer-synccheck
```

## Implementation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p cuda_kernel_sandbox/kernels
```

### Step 2: Create `cuda_compat.cuh`
HIP-to-CUDA mapping macros.

### Step 3: Create `rocsolver_types.cuh`
Minimal type definitions extracted from rocBLAS/rocSOLVER headers.

### Step 4: Create `device_helpers.cuh`
Device helper functions: `aabs`, `swap`, `load_ptr_batch`.

### Step 5: Create `kernels/getf2_small_kernel.cuh`
Copy the `getf2_small_kernel` template from `library/src/specialized/roclapack_getf2_specialized_kernels.hpp`:
- Lines 25-140 contain the kernel
- **Remove the `__syncthreads()` on line 103**
- Update includes to use sandbox headers

### Step 6: Create `sandbox_getf2_small.cu`
Main driver with:
- Command-line argument parsing for m, n, batch_count
- Random matrix initialization
- Grid/block dimension calculation (mirroring `getf2_run_small` logic)
- Kernel launch with template dispatch
- Result verification

### Step 7: Create `Makefile`
Build system with sanitizer targets.

## Usage

### Build
```bash
cd cuda_kernel_sandbox
make
```

### Run with CUDA Compute Sanitizer
```bash
# Memory error detection
make sanitizer-memcheck

# Race condition detection (what we're looking for)
make sanitizer-racecheck

# Uninitialized memory detection
make sanitizer-initcheck

# Synchronization errors
make sanitizer-synccheck
```

### Custom Parameters
```bash
./sandbox_getf2_small 32 16 4    # m=32, n=16, batch=4
```

## Extending to Other Kernels

To test a different kernel:

1. Copy the kernel to `kernels/<kernel_name>.cuh`
2. Modify includes to use sandbox headers
3. Create `sandbox_<kernel_name>.cu` with appropriate:
   - Test data initialization
   - Grid/block configuration
   - Template instantiation
4. Add build target to Makefile

## Notes

- The `-g -G -lineinfo` flags enable debug info for better sanitizer reports
- For racecheck to detect the missing `__syncthreads()`, ensure the kernel is run with appropriate parameters that exercise the race condition
- Complex number support is stubbed but functional if needed
- This is intentionally minimal - add types/functions as needed for other kernels
