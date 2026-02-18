# rocSOLVER CUDA Kernel Sandbox

A minimal test harness for compiling individual rocSOLVER kernels with nvcc to use CUDA Compute Sanitizer tools.

## Quick Start

```bash
# Build
make

# Run racecheck (detects missing __syncthreads)
make sanitizer-racecheck

# Run with custom parameters
make sanitizer-custom M=32 N=16 BATCH=4
```

## Current Kernel: getf2_small_kernel

The `getf2_small_kernel` has been modified to **remove the `__syncthreads()` call** that should synchronize threads before overwriting shared memory. This allows testing whether CUDA Compute Sanitizer's racecheck tool can detect this race condition.

## Files

- `cuda_compat.cuh` - HIP-to-CUDA compatibility macros
- `rocsolver_types.cuh` - Minimal type definitions
- `device_helpers.cuh` - Device helper functions (aabs, swap, load_ptr_batch)
- `kernels/getf2_small_kernel.cuh` - The kernel under test
- `sandbox_getf2_small.cu` - Main driver program
- `Makefile` - Build system

## Adding New Kernels

1. Copy the kernel to `kernels/<kernel_name>.cuh`
2. Update includes to use sandbox headers
3. Create `sandbox_<kernel_name>.cu` with test setup
4. Add build target to Makefile

## Sanitizer Tools

- **racecheck**: Detects shared memory race conditions (missing `__syncthreads`)
- **memcheck**: Detects memory errors (out of bounds, misaligned access)
- **initcheck**: Detects use of uninitialized memory
- **synccheck**: Detects synchronization errors
