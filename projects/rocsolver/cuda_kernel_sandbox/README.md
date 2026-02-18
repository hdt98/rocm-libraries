# rocSOLVER CUDA Kernel Sandbox

A minimal test harness for compiling individual rocSOLVER kernels with nvcc to use CUDA Compute Sanitizer tools.

## Quick Start

```bash
# Build all sandboxes
make

# Run racecheck on getf2_small (detects missing __syncthreads)
make sanitizer-racecheck

# Run racecheck on stebz
make stebz-racecheck

# Run with custom parameters
make sanitizer-custom M=32 N=16 BATCH=4
make stebz-racecheck STEBZ_N=128 STEBZ_BATCH=4
```

## Available Kernels

### 1. getf2_small_kernel (LU Factorization)

Small LU factorization kernel for matrices where m >= n and n <= 64.

The `getf2_small_kernel` has been modified to **remove the `__syncthreads()` call** that should synchronize threads before overwriting shared memory. This allows testing whether CUDA Compute Sanitizer's racecheck tool can detect this race condition.

```bash
# Build
make sandbox_getf2_small

# Run
./sandbox_getf2_small [m] [n] [batch_count]

# Run with sanitizer
make sanitizer-racecheck
```

### 2. stebz_kernels (Eigenvalue Bisection)

Eigenvalue computation for symmetric tridiagonal matrices using Sturm sequence and bisection method.

Includes three kernels:
- `stebz_splitting_kernel` - Splits matrix into independent blocks
- `stebz_bisection_kernel` - Iterative bisection to find eigenvalues
- `stebz_synthesis_kernel` - Synthesizes results from split blocks

```bash
# Build
make sandbox_stebz

# Run
./sandbox_stebz [n] [batch_count] [range]
# range: all, value, or index

# Run with sanitizer
make stebz-racecheck
make stebz-racecheck STEBZ_N=128 STEBZ_BATCH=4 STEBZ_RANGE=all
```

## Files

### Common Files
- `cuda_compat.cuh` - HIP-to-CUDA compatibility macros
- `rocsolver_types.cuh` - Minimal type definitions (rocblas_stride, enums, etc.)
- `device_helpers.cuh` - Device helper functions (aabs, swap, load_ptr_batch, iamax)
- `Makefile` - Build system

### getf2_small
- `kernels/getf2_small_kernel.cuh` - LU factorization kernel
- `sandbox_getf2_small.cu` - Driver program

### stebz
- `kernels/stebz_kernels.cuh` - Eigenvalue bisection kernels
- `sandbox_stebz.cu` - Driver program

## Adding New Kernels

1. Copy the kernel to `kernels/<kernel_name>.cuh`
2. Update includes to use sandbox headers:
   ```cpp
   #include "../cuda_compat.cuh"
   #include "../rocsolver_types.cuh"
   #include "../device_helpers.cuh"
   ```
3. Create `sandbox_<kernel_name>.cu` with test setup
4. Add build target to Makefile

## Sanitizer Tools

- **racecheck**: Detects shared memory race conditions (missing `__syncthreads`)
- **memcheck**: Detects memory errors (out of bounds, misaligned access)
- **initcheck**: Detects use of uninitialized memory
- **synccheck**: Detects synchronization errors

## Make Targets

### Build
- `make all` - Build all sandbox executables
- `make sandbox_getf2_small` - Build getf2_small sandbox only
- `make sandbox_stebz` - Build stebz sandbox only
- `make clean` - Remove built executables

### getf2_small Sanitizer
- `make sanitizer-memcheck`
- `make sanitizer-racecheck`
- `make sanitizer-racecheck-all`
- `make sanitizer-initcheck`
- `make sanitizer-synccheck`
- `make sanitizer-all`

### STEBZ Sanitizer
- `make run-stebz` - Run without sanitizer
- `make stebz-memcheck`
- `make stebz-racecheck`
- `make stebz-racecheck-all`
- `make stebz-initcheck`
- `make stebz-synccheck`
- `make stebz-all`

### Custom Parameters
```bash
# getf2_small
make test-custom M=32 N=16 BATCH=4
make sanitizer-custom M=32 N=16 BATCH=4

# stebz
make run-stebz STEBZ_N=128 STEBZ_BATCH=4 STEBZ_RANGE=all
make stebz-racecheck STEBZ_N=128 STEBZ_BATCH=4 STEBZ_RANGE=value
```
