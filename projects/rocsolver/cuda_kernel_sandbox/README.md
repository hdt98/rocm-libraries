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

# Run racecheck on stedc solve (slaed4)
make stedc-racecheck

# Run racecheck on potf2 (Cholesky)
make potf2-racecheck

# Run with custom parameters
make sanitizer-custom M=32 N=16 BATCH=4
make stebz-racecheck STEBZ_N=128 STEBZ_BATCH=4
make stedc-racecheck STEDC_N=32 STEDC_REF=1
make potf2-racecheck POTF2_N=32 POTF2_BATCH=4 POTF2_UPLO=lower
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

### 3. stedc_solve_kernels (Secular Equation Solvers)

Secular equation solvers used in the merge phase of divide-and-conquer eigenvalue computation (STEDC). These kernels compute eigenvalues of a rank-1 perturbation of a diagonal matrix: D + rho * z * z^T.

Includes:
- `slaed6` - Solves modified secular equation using 3 poles (Gragg-Thornton-Warner scheme)
- `slaed4` - Reference LAPACK-style solver, calls slaed6 for 3-pole interpolation
- `seq_eval` - Evaluates secular equation at a point
- `seq_solve` - Optimized solver for internal eigenvalues
- `seq_solve_ext` - Optimized solver for the last eigenvalue
- `stedc_mergeValues_Solve_kernel` - Main kernel that calls the solvers

```bash
# Build
make sandbox_stedc_solve

# Run
./sandbox_stedc_solve [n] [batch_count] [use_reference]
# use_reference: 1 for slaed4, 0 for seq_solve

# Run with sanitizer
make stedc-racecheck
make stedc-racecheck STEDC_N=32 STEDC_BATCH=4 STEDC_REF=1
```

### 4. potf2_kernels (Cholesky Factorization)

Cholesky factorization for symmetric positive definite matrices. Computes A = L*L' (lower) or A = U'*U (upper).

Includes:
- `potf2_kernel_small` - Shared memory kernel for small matrices (n <= 64)
- `potf2_simple` - Device function performing the factorization
- `sqrtDiagOnward` - Diagonal element processing kernel
- `potf2_unrolled_kernel` - Alternative unrolled implementation

```bash
# Build
make sandbox_potf2

# Run
./sandbox_potf2 [n] [batch_count] [uplo]
# uplo: lower or upper

# Run with sanitizer
make potf2-racecheck
make potf2-racecheck POTF2_N=32 POTF2_BATCH=4 POTF2_UPLO=lower
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

### stedc_solve
- `kernels/stedc_solve_kernels.cuh` - Secular equation solvers (slaed4, slaed6, seq_solve)
- `sandbox_stedc_solve.cu` - Driver program

### potf2
- `kernels/potf2_kernels.cuh` - Cholesky factorization kernels
- `sandbox_potf2.cu` - Driver program

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
- `make sandbox_stedc_solve` - Build stedc solve sandbox only
- `make sandbox_potf2` - Build potf2 sandbox only
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

### STEDC Solve Sanitizer
- `make run-stedc-solve` - Run without sanitizer
- `make stedc-memcheck`
- `make stedc-racecheck`
- `make stedc-racecheck-all`
- `make stedc-initcheck`
- `make stedc-synccheck`
- `make stedc-all`

### POTF2 (Cholesky) Sanitizer
- `make run-potf2` - Run without sanitizer
- `make potf2-memcheck`
- `make potf2-racecheck`
- `make potf2-racecheck-all`
- `make potf2-initcheck`
- `make potf2-synccheck`
- `make potf2-all`

### Custom Parameters
```bash
# getf2_small
make test-custom M=32 N=16 BATCH=4
make sanitizer-custom M=32 N=16 BATCH=4

# stebz
make run-stebz STEBZ_N=128 STEBZ_BATCH=4 STEBZ_RANGE=all
make stebz-racecheck STEBZ_N=128 STEBZ_BATCH=4 STEBZ_RANGE=value

# stedc_solve
make run-stedc-solve STEDC_N=32 STEDC_BATCH=4 STEDC_REF=1
make stedc-racecheck STEDC_N=16 STEDC_REF=0

# potf2
make run-potf2 POTF2_N=32 POTF2_BATCH=4 POTF2_UPLO=lower
make potf2-racecheck POTF2_N=64 POTF2_UPLO=upper
```
