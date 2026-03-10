# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **rocm-libraries monorepo** — a consolidation of AMD's ROCm library ecosystem. The primary project being actively developed here is **Composable Kernel (CK)**, located at `projects/composablekernel/`. CK is a GPU kernel library providing performance-critical ML workloads via HIP C++, using a tile-based programming model and Tensor Coordinate Transformation.

## Development Environment (enroot)

All build and test work runs inside an **enroot** container that provides the ROCm toolchain. The container image is at `~/enroot/image_ck_dev.sqsh`.

### First-time setup

```bash
# Check if the container already exists
enroot list -f

# If it does not appear in the list, create it (one-time)
enroot create --name ck_dev_env ~/enroot/image_ck_dev.sqsh
```

> **Note:** The container name used when creating (`--name`) must match the name used in `enroot start`. The image ships as `image_ck_dev.sqsh`; the container is conventionally named `ck_dev_env`.

### Starting an interactive shell

Run this from the **repository root** (the `--mount` path must be an absolute path and the same inside and outside the container):

```bash
enroot start --rw --mount ${PWD}:${PWD} ck_dev_env sh -c "cd ${PWD} && bash"
```

All subsequent build commands (`cmake`, `make`, `ctest`, `ckProfiler`) should be run inside this shell.

### Running a one-off command without an interactive shell

```bash
REPO=/path/to/rocm-libraries-ck2
enroot start --rw --mount ${REPO}:${REPO} ck_dev_env sh -c "cd ${REPO}/projects/composablekernel/build && ninja -j32 ckProfiler"
```

### Verified example: cmake configure for gfx950

From inside an enroot shell (or as a one-off command), create a build directory and configure:

```bash
cd projects/composablekernel
mkdir build-gfx950 && cd build-gfx950
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS=gfx950 \
  -G Ninja \
  ..
```

Expected output highlights (configure takes ~3 minutes):
- Compiler: `Clang 20.0.0` via `/opt/rocm/bin/hipcc`
- `Building CK for the following targets: gfx950`
- `Enabling XDL instances` and `Enabling XDL FP8 gemms on gfx950`
- `Configuring done` — ready to `ninja -j32`

## Build Commands

### Composable Kernel (primary project)

Build from a dedicated build directory inside `projects/composablekernel/`:

```bash
mkdir build && cd build

# Development build (convenience script — cleans CMake cache, sets sensible defaults)
../script/cmake-ck-dev.sh                        # defaults to gfx908;gfx90a;gfx942
../script/cmake-ck-dev.sh .. gfx942             # single target
../script/cmake-ck-dev.sh .. gfx90a -DCMAKE_BUILD_TYPE=Release

# Manual cmake configure
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx950" \
  -G Ninja \
  ..

# Build (limit parallelism to avoid OOM — each thread uses ~2GB RAM)
ninja -j64

# Specific targets
ninja -j ckProfiler
ninja -j examples tests
ninja -j check          # build + run all tests
ninja -j smoke          # tests < 30s each
ninja -j regression     # tests >= 30s each
```

**Key CMake flags to speed up builds:**
- `DTYPES="fp16;fp32"` — limit data types (default builds all: fp64;fp32;tf32;fp16;fp8;bf16;int8)
- `DISABLE_DL_KERNELS=ON` — skip DL kernel instances
- `DISABLE_DPP_KERNELS=ON` — skip DPP kernel instances
- `CK_PROFILER_OP_FILTER="grouped_gemm"` — build only matching profiler ops (regex)
- `CK_PROFILER_INSTANCE_FILTER="..."` — filter kernel instances (regex)

**Note:** Tests and examples are only built if `GPU_TARGETS` is set explicitly on the cmake command line.

**sccache** can reduce rebuild time from hours to minutes:
```bash
sccache --start-server
# Then add to cmake: -DCMAKE_HIP_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
```

### Superbuild (all components)

```bash
cmake -B build -S . -D CMAKE_INSTALL_PREFIX=/opt/rocm -D CMAKE_PREFIX_PATH=/opt/rocm
cmake --build build

# Using presets (see CMakePresets.json for available presets)
cmake --list-presets=configure
cmake --preset rocroller
cmake --build --preset default
```

## Testing

```bash
# Run all tests via CTest (from build directory)
ctest

# Run a single test binary directly
./bin/<test_name>

# Run via make targets
ninja -j smoke        # fast tests (<30s)
ninja -j regression   # slow tests (>=30s)
```

Tests live in `projects/composablekernel/test/` (68+ subdirectories covering GEMM, convolution, attention, normalization, and CK Tile operations).

## Profiling

```bash
# Profile GEMM (fp16, NT layout, verify, time kernel, 4096x4096x4096)
./bin/ckProfiler gemm_universal 1 0 1 1 0 1 4096 4096 4096 4096 4096 4096 1 1 10 0

# Profile 2D forward convolution
./bin/ckProfiler conv2d_fwd 1 1 1 1 1 1 0 5 128 256 192 3 3 71 71 2 2 1 1 1 1 1 1

# List all available profiler ops
find profiler/src -name "profile_*.cpp" | sed 's|profiler/src/profile_||' | sed 's|.cpp||' | sort

# Convert MIOpen driver command to profiler command
python3 ../script/convert_miopen_driver_to_profiler.py
```

## Code Formatting

CK uses `clang-format`. Install pre-commit hooks to auto-format on commit:

```bash
# From projects/composablekernel/
sudo script/install_precommit.sh

# Or from monorepo root
pip install pre-commit
pre-commit install

# Run manually on staged files
pre-commit

# Run on all files in a project
pre-commit run --files $(git ls-files projects/composablekernel)
```

Note: `composablekernel` is currently **excluded** from the monorepo-level pre-commit checks. The project has its own pre-commit configuration.

## Architecture

### CK Library Layers (bottom to top)

1. **Templated Tile Operators** (`include/ck/`) — Low-level GPU primitives. Tensor coordinate transformations live here, enabling complex ML operators to be decomposed into tile operations.
2. **Templated Kernel and Invoker** — Generic kernel templates parameterized over tile operators.
3. **Instantiated Kernel and Invoker** (`library/src/`) — Concrete specializations for specific data types, layouts, and GPU targets. Generated by `codegen/`.
4. **Client API** (`client_example/`, `dispatcher/`) — High-level interfaces used by MIOpen, hipBLAS, etc.

### Key Header Directories

- `include/ck/` — Classic CK engine
  - `tensor_operation/` — Operation implementations (GEMM, conv, attention)
  - `tensor_description/` — Tensor descriptor and coordinate transformation primitives
  - `utility/` — AMD GPU architecture support, math utilities
  - `problem_transform/` — Transforms convolution problems into GEMM
- `include/ck_tile/` — **Modern CK Tile engine** (preferred for new kernel development)
  - `core/` — Tile primitives (load/store, MMA wrappers)
  - `ops/` — Higher-level ops (GEMM, FMHA, GroupNorm, LayerNorm)
  - `host/` — Host-side kernel launch utilities

### Tensor Coordinate Transformation

The core innovation of CK. Multi-dimensional tensors are described as coordinate systems with transformations (Merge, Unmerge, Pad, Slice, etc.) that compose to map logical indices to physical memory addresses. This allows a single kernel template to handle arbitrary layouts, strides, and problem shapes without code duplication. Transformations are defined in `include/ck/tensor_description/`.

### Codegen

`codegen/` contains Python scripts that generate C++ kernel instantiation files based on a configuration system. When adding new operation instances, the codegen must be updated and re-run (CMake handles this automatically during build).

### CK Tile vs. Classic CK

New operations should be implemented in `include/ck_tile/` (CK Tile), which has a cleaner API and better composability. The classic `include/ck/` headers remain for backward compatibility. CK Tile tests live in `test/ck_tile/`.

### GPU Target Architecture Support

- **gfx908** (MI100) — MFMA, no native fp8
- **gfx90a** (MI200) — MFMA, no native fp8
- **gfx942** (MI300) — MFMA, native fp8
- **gfx950** (MI350) — MFMA, native fp8/fp4
- **gfx11xx** (RDNA3) — WMMA instead of XDL/MFMA

Use `GPU_TARGETS` for similar arch families; use `GPU_ARCHS` for cross-family builds (e.g., `GPU_ARCHS=gfx908;gfx1100;gfx942`).

## Branching Conventions

- Branch naming: `users/<github-username>/ck/<descriptive-name>`
- Main branch: `develop`
- Keep history linear: use `git rebase origin/develop` (not merge)
- PRs from branches within the repo have fuller CI privileges than forks
