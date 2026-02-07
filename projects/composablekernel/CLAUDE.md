# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Composable Kernel (CK) is a performance-critical kernel library for machine learning workloads on AMD GPUs using HIP C++. It provides:
- Tile-based programming model for GPU kernels
- Tensor Coordinate Transformation for complex layout/index operations
- Pre-built kernel instances for operations like GEMM, convolutions, attention, normalization, etc.
- Support for multiple AMD GPU architectures (gfx908, gfx90a, gfx942, gfx950)

## Build Commands

### Standard Build

```bash
# From repository root
mkdir -p build && cd build

# Configure with GPU targets (REQUIRED)
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx908;gfx90a;gfx942" \
  ..

# Build library
make -j$(nproc)

# Install library
make install
```

**Important GPU Target Notes:**
- You MUST set `GPU_TARGETS` on the cmake command line. If not set, CK builds for ALL architectures (very slow).
- Tests and examples only build if `GPU_TARGETS` is explicitly set.
- For similar architectures, use `GPU_TARGETS` (e.g., `gfx908;gfx90a`).
- For different architectures, use `GPU_ARCHS` instead (e.g., `GPU_ARCHS=gfx908;gfx1030;gfx942`).

### Development Build (Recommended)

```bash
# From repository root
mkdir -p build && cd build

# Use convenience script (defaults to gfx908;gfx90a;gfx942)
../script/cmake-ck-dev.sh

# Or specify custom GPU targets
../script/cmake-ck-dev.sh .. gfx90a

# Or with additional cmake args
../script/cmake-ck-dev.sh .. gfx90a -DCMAKE_BUILD_TYPE=Debug

# Build
make -j$(nproc)
```

The dev script enables `BUILD_DEV=ON`, verbose makefiles, and better error messages.

### CMake Presets

```bash
# Use preset for specific GPU (from build directory)
cmake .. --preset dev-gfx90a
make -j$(nproc)
```

Available presets: `dev`, `dev-gfx908`, `dev-gfx90a`, `dev-gfx942`, `dev-gfx950`

### Build Optimization Flags

To speed up builds, use these CMake options:

```bash
# Build only specific data types (default builds all)
-D DTYPES="fp32;fp16"  # Options: fp64;fp32;tf32;fp16;fp8;bf16;int8

# Skip DL kernels (useful if targeting non-NAVI2x)
-D DISABLE_DL_KERNELS=ON

# Skip DPP kernels
-D DISABLE_DPP_KERNELS=ON

# Enable FP8 on unsupported architectures (gfx908, gfx90a) for functional testing
-D CK_USE_FP8_ON_UNSUPPORTED_ARCH=ON
```

### Using sccache

```bash
# Start sccache server
sccache --start-server

# Add to cmake command
cmake \
  -D CMAKE_HIP_COMPILER_LAUNCHER=sccache \
  -D CMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -D CMAKE_C_COMPILER_LAUNCHER=sccache \
  [other options] \
  ..
```

## Testing

### Run All Tests

```bash
# Build and run all tests and examples
make -j check

# Build only smoke tests (< 30 seconds each)
make -j smoke

# Build only regression tests (>= 30 seconds each)
make -j regression

# Build tests without running
make -j tests

# Build examples without running
make -j examples
```

### Run Individual Tests

```bash
# From build directory
# Tests are located in build/test/<operation>/
./test/gemm/test_gemm
./test/ck_tile/fmha/test_fmha_fwd

# Examples are in build/example/<name>/
./example/01_gemm/example_gemm
./example/ck_tile/01_fmha/example_fmha
```

## Profiler

### Build Profiler

```bash
# Build all profilers
make -j ckProfiler

# Build specific profiler operations (use regex)
cmake -D CK_PROFILER_OP_FILTER="grouped_gemm" .. && make -j

# Build ONLY exact operation (use ^$ anchors)
cmake -D CK_PROFILER_OP_FILTER="^grouped_gemm$" .. && make -j
```

### Run Profiler Examples

```bash
# From build/bin/
# GEMM profiler
./ckProfiler gemm_universal 1 0 1 1 0 1 4096 4096 4096 4096 4096 4096 1

# Grouped convolution forward
./ckProfiler grouped_conv_fwd 1 0 1 1 0 1 2 32 4 192 192 3 3 28 28 1 1 1 1 1 1 1 1
```

See `profiler/README.md` for complete profiler argument documentation.

## Client Examples

Client examples show how to use CK as a pre-built library (not building kernels from scratch).

```bash
# Build client examples (CK must be installed first)
mkdir -p client_example/build && cd client_example/build

cmake \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm" \
  -D GPU_TARGETS="gfx908;gfx90a" \
  ..

make -j
```

## Architecture

### Two Programming Models

CK provides two distinct programming interfaces:

1. **CK (Legacy)** - Located in `include/ck/`
   - Original tile-based programming model
   - Kernel templates and device operations
   - Extensive pre-built kernel instances in `library/src/tensor_operation_instance/gpu/`

2. **CK_Tile (Modern)** - Located in `include/ck_tile/`
   - Modern, cleaner programming model
   - Component-based architecture with single-header includes
   - Independent from legacy CK (no dependencies on `include/ck/`)
   - Recommended for new development

### CK_Tile Component Structure

`include/ck_tile/` is organized into:

- **`core/`** - Basic building blocks (include `ck_tile/core.hpp`)
  - `container/` - array, tuple, sequence
  - `numeric/` - GPU data types (fp16_t, bf16_t, fp8_t), conversions
  - `algorithm/` - Coordinate transformation system (merge, unmerge, embed)
  - `tensor/` - Tensor descriptors, distributed tensors, tile-level APIs

- **`host/`** - Host utilities (kernel launch, device buffer management)
  - Use specific headers (e.g., `ck_tile/host/kernel_launch.hpp`)

- **`ops/`** - Device operator implementations
  - `ops/gemm/`, `ops/fmha/`, `ops/reduce/`, etc.
  - Each contains: warp, block, pipeline, kernel templates
  - Include specific headers: `ck_tile/ops/fmha.hpp`, `ck_tile/ops/gemm.hpp`

- **`ops/epilogue/`** - Epilogue implementations for kernel fusion

### Directory Structure

```
include/
├── ck/              # Legacy CK headers
├── ck_tile/         # Modern CK_Tile headers
└── rapidjson/       # JSON parsing

library/
└── src/
    ├── tensor_operation_instance/gpu/  # Pre-compiled kernel instances
    │   ├── gemm/, batched_gemm/, grouped_gemm/
    │   ├── convolution variants
    │   ├── normalization/, softmax/, reduce/
    │   └── mha/ (multi-head attention)
    └── utility/

example/             # Examples building kernels from templates
├── 01_gemm/, 15_grouped_gemm/, etc.
└── ck_tile/         # CK_Tile examples
    ├── 01_fmha/, 05_reduce/, 19_gemm_multi_d/, etc.

client_example/      # Examples using pre-built library (must install first)
├── 01_gemm/, 07_grouped_convnd_fwd/, etc.

test/                # Tests for each operation
├── gemm/, batched_gemm/, grouped_gemm/
├── convnd_fwd/, grouped_convnd_bwd_data/
└── ck_tile/         # CK_Tile tests

profiler/            # Performance profiling tools

dispatcher/          # Unified kernel dispatch system (C++ and Python)
                    # Validated for MI300 series (gfx942)

codegen/             # Code generation utilities
```

### Instance Generation

Pre-built kernel instances are in `library/src/tensor_operation_instance/gpu/<operation>/`.
Each operation directory contains multiple files with different kernel configurations (XDL, WMMA, DL variants).

## Key Concepts

### Tensor Coordinate Transformation
Core abstraction for layout/index transforms at compile-time and runtime. Describes how ND tensors are built using primitives like `merge`, `unmerge`, `embed`.

### Tile Programming Model
Operations work on tiles (sub-tensors) distributed across threads/warps/blocks:
- **Distributed Tensor** - Describes tensor storage and thread collaboration
- **Load/Store Tile** - Tile-level data movement APIs
- **Block Tile** - Tile processed by a thread block
- **Wave Tile** - Tile processed by a wavefront (warp)

### Matrix Instructions
- **MFMA** - Matrix Fused Multiply-Add (AMD GPU instruction)
- **XDL** - AMD's high-performance matrix multiply instructions
- **WMMA** - Warp Matrix Multiply-Accumulate
- **DL** - Deep Learning instructions (useful on NAVI2x)

## Code Style

### Pre-commit Hooks

```bash
# Install pre-commit hooks (runs clang-format before commits)
sudo script/install_precommit.sh

# Uninstall hooks
script/uninstall_precommit.sh

# Skip hooks for a commit (use sparingly)
git commit --no-verify
```

### Manual Formatting

```bash
# Format all changed files
clang-format -i path/to/file.hpp
```

## Reference Documentation

- Main docs: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
- `TERMINOLOGY.md` - Technical glossary (hardware, operations, programming model)
- `ACRONYMS.md` - Common acronyms (GEMM, FMHA, MFMA, etc.)
- `include/ck/BUILD_TIME_OPTIMIZATION.md` - Build optimization strategies
- Individual example READMEs in `example/` and `client_example/` subdirectories

## Common Development Patterns

### Adding a New Kernel Instance

1. Locate the operation in `library/src/tensor_operation_instance/gpu/<operation>/`
2. Add new instance configuration to appropriate file (e.g., `device_gemm_xdl_*.cpp`)
3. Rebuild library

### Creating a New Example

1. For legacy CK: Add to `example/`
2. For CK_Tile: Add to `example/ck_tile/`
3. Create CMakeLists.txt with proper dependencies
4. Follow existing examples for structure

### Working with CK_Tile

When writing new kernels with CK_Tile:
- Include only needed headers: `#include "ck_tile/core.hpp"`, `#include "ck_tile/ops/gemm.hpp"`
- Use component-based design (warp → block → pipeline → kernel)
- Reference examples in `example/ck_tile/` for patterns
- Do NOT include anything from `include/ck/` in CK_Tile code

## Important Notes

- **Never commit without clang-format** - Pre-commit hooks enforce this
- **Always set GPU_TARGETS** - Builds without it take hours
- **Limit parallel jobs** - Each compilation thread uses ~2GB RAM. On high-core systems with limited RAM, use `-j32` or similar instead of `-j$(nproc)`
- **XDL vs DL kernels** - XDL is preferred except on NAVI2x where DL may be faster
- **CK vs CK_Tile** - Prefer CK_Tile for new development; it's cleaner and independent
