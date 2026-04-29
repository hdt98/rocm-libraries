# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Composable Kernel (CK) is a high-performance kernel library for machine learning workloads on AMD GPUs, written in HIP C++. It provides a tile-based programming model and uses Tensor Coordinate Transformation for performance portability. CK lives at `projects/composablekernel/` within the rocm-libraries monorepo.

## Build Commands

CK requires `GPU_TARGETS` to be set (otherwise builds for all targets, which is very slow). Each build thread uses ~2GB RAM — limit parallelism accordingly.

```bash
cd projects/composablekernel
mkdir build && cd build

# Standard build
cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  ..
make -j$(nproc)

# Dev build with convenience script (cleans cmake cache, sets BUILD_DEV=ON)
../script/cmake-ck-dev.sh .. gfx942
make -j$(nproc)

# Minimal dev build (~5s configure vs ~150s; no instances/profiler/examples/tests)
../script/cmake-ck-dev.sh --minimal .. gfx942

# Using CMake presets
cmake --preset dev-gfx942 ..
```

### Key CMake Options

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_TARGETS` | all supported | Target GPU architectures (e.g. `gfx908;gfx90a;gfx942`) |
| `GPU_ARCHS` | unset | Alternative to GPU_TARGETS for library-only builds (no tests/examples) |
| `DTYPES` | all | Filter data types: `fp64;fp32;tf32;fp16;fp8;bf16;int8` |
| `BUILD_DEV` | ON | Enable `-Werror -Weverything` |
| `BUILD_CK_DEVICE_INSTANCES` | ON | Build device operation instances in library/ |
| `BUILD_CK_PROFILER` | ON | Build ckProfiler |
| `BUILD_CK_EXAMPLES` | ON | Build examples |
| `BUILD_CK_TILE_ENGINE` | ON | Build tile_engine |
| `DISABLE_DL_KERNELS` | OFF | Skip DL instances (only useful on NAVI2x) |
| `CK_USE_CODEGEN` | OFF | Enable codegen library |
| `CK_CXX_STANDARD` | 20 | C++ standard (17 or 20) |
| `MIOPEN_REQ_LIBS_ONLY` | OFF | Build only MIOpen-required libraries |
| `CK_PROFILER_OP_FILTER` | all | Regex to filter which profiler operations to build |

### Build Speed Tips

- Use `DTYPES="fp16;fp32"` to skip unnecessary data types
- Use `--minimal` flag with cmake-ck-dev.sh for header-only development
- Use `sccache` for incremental rebuilds (pre-installed in CK Docker images)
- Use `CK_PARALLEL_LINK_JOBS` and `CK_PARALLEL_COMPILE_JOBS` with Ninja to limit resource usage

## Testing

Tests and examples only build when `GPU_TARGETS` is explicitly set by the user.

```bash
# Build and run all tests+examples
make -j check

# Smoke tests (each < 30 seconds)
make -j smoke

# Regression tests (each >= 30 seconds)
make -j regression

# Build just tests or examples
make -j tests
make -j examples
```

## Architecture

### Four-Layer Design

1. **Templated Tile Operators** — Low-level tile operations (load, store, GEMM, shuffle)
2. **Templated Kernel and Invoker** — Kernel templates composed from tile operators
3. **Instantiated Kernel and Invoker** — Concrete instantiations for specific types/layouts
4. **Client API** — High-level interface for end users

### Directory Layout

| Directory | Purpose |
|-----------|---------|
| `include/ck/` | Legacy CK headers: tensor operations, utilities, wrapper API |
| `include/ck_tile/` | New tile-based API (independent from legacy CK): core, host, ops, ref |
| `library/` | Pre-built device operation instances (`library/src/tensor_operation_instance/gpu/`) |
| `profiler/` | ckProfiler — performance profiling tool for all operations |
| `example/` | Standalone examples using templated kernel/invoker directly |
| `client_example/` | Examples using the instance factory (client API) |
| `test/` | Tests organized by operation (batched_gemm, convnd_fwd, gemm, etc.) |
| `codegen/` | Code generation for kernel instances |
| `dispatcher/` | Unified kernel dispatch system with C++ and Python frontends |
| `tile_engine/` | Tile engine with operation support matrix |
| `script/` | Build scripts, profiling scripts, infrastructure tools |

### CK vs CK Tile

- **`ck`** (`include/ck/`): The original API. Uses tensor coordinate transformations.
- **`ck_tile`** (`include/ck_tile/`): The newer tile-based API, independently usable. Include single headers per component (e.g., `#include "ck_tile/core.hpp"`, `#include "ck_tile/ops/fmha.hpp"`). Transitioning to replace the legacy API.

### Key Operations

GEMM (including batched, grouped, strided, universal, streamk), convolution (forward, backward data, backward weight), attention/FMHA, batchnorm, layernorm, groupnorm, softmax, reduce, pooling, permute, elementwise operations.

## Pre-commit

```bash
sudo script/install_precommit.sh   # install hooks
# Uses clang-format per .clang-format
```

## Profiler

```bash
# Build specific profiler operations only (regex filter)
cmake -DCK_PROFILER_OP_FILTER="grouped_gemm" ...

# Run profiler
./bin/ckProfiler gemm_universal 1 0 1 1 0 1 4096 4096 4096 4096 4096 4096 1
```
