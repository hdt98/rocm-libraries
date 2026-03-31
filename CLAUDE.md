# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo (`rocm-libraries`) hosting multiple ROCm GPU computing libraries. The primary focus for this checkout is **ComposableKernel (CK)**, located at `projects/composablekernel/`. Other libraries (`projects/hipblas`, `projects/miopen`, `shared/rocroller`, etc.) exist but are generally out-of-scope for CK-focused work.

## Building ComposableKernel

**Manual cmake (for custom flags):**
```bash
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx950" \
  -D DTYPES="bf16" \
  -G NInja
  ..
ninja -j$(nproc)
```

**Key CMake flags:**
- `GPU_TARGETS` — semicolon-separated GPU arch list (e.g., `gfx908;gfx90a;gfx942;gfx950`)
- `DTYPES` — limit compiled data types to reduce build time (e.g., `"fp16;bf16"`)
- `CK_PROFILER_OP_FILTER` — regex to filter profiler operations compiled in
- `CK_PROFILER_INSTANCE_FILTER` — regex to filter kernel instances compiled in
- `DISABLE_DL_KERNELS`, `DISABLE_DPP_KERNELS` — skip specific kernel families
- `CK_CXX_STANDARD` — 17 or 20 (default: 20)

**sccache for faster rebuilds:**
```bash
sccache --start-server
cmake -DCMAKE_HIP_COMPILER_LAUNCHER=sccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
      -DCMAKE_C_COMPILER_LAUNCHER=sccache ..
```

## Running Tests

The test binaries are located inside the build directory `projects/composablekernel/build-gfx950` (or the generic `projects/composablekernel/build`)

# Run a specific test binary directly
./bin/test_gemm_universal

# Filter via ctest
ctest -R conv2d_fwd --verbose
```

Tests are organized under `projects/composablekernel/test/` by operation (e.g., `convnd_fwd/`, `gemm/`, `grouped_convnd_fwd/`).

## Building the Profiler

The CK Profiler binaries are located inside the build directory `projects/composablekernel/build-gfx950` (or the generic `projects/composablekernel/build`)

```bash
ninja -j$(nproc) ckProfiler

# Profile grouped forward convolution
./bin/ckProfiler profile_grouped_conv_fwd ...

# Profile 2D convolution
./bin/ckProfiler conv2d_fwd 1 1 1 1 1 1 0 5 128 256 192 3 3 71 71 2 2 1 1 1 1 1 1
```

The profiler's `CMakeLists.txt` (`profiler/src/CMakeLists.txt`) controls which operations and kernel instances are compiled. Use `CK_PROFILER_OP_FILTER` / `CK_PROFILER_INSTANCE_FILTER` cmake flags to narrow compilation.

## Architecture

CK uses a **four-layer template metaprogramming model**:

1. **Templated Tile Operators** (`include/ck/`, `include/ck_tile/`) — Low-level tile load/store/compute primitives; hardware-aware, architecture-specific.
2. **Templated Kernel & Invoker** (`include/ck/tensor_operation/`) — Compose tile operators into complete GPU kernels via policy-based templates.
3. **Instantiated Kernel & Invoker** (`library/src/`) — Fully specialized kernel instances for specific shapes, data types, and hardware. These are the files that take long to compile.
4. **Client API / Instance Factory** (`library/include/`, `client_example/`) — Runtime kernel selection and user-facing dispatch.

**Key directories:**
- `include/ck/` — Core headers (tensor, utility, tensor_operation)
- `include/ck_tile/` — Newer tile-based API (tile_program, ops, host)
- `library/src/tensor_operation_instance/` — Instantiated kernels for each op
- `example/` — 69+ standalone examples per operation type
- `client_example/` — Examples using the instance-factory pattern
- `profiler/` — `ckProfiler` benchmarking tool
- `test/` — CTest-based test suite
- `codegen/` — Python-based code generation
- `dispatcher/` — Dynamic kernel selection with Python bindings
- `python/` — `ck4inductor` PyTorch integration

**GPU architecture targets:**
- `gfx908` (MI100), `gfx90a` (MI200), `gfx942` (MI300), `gfx950` (MI350)
- `gfx1030`/`gfx1100`/`gfx1101`/`gfx1102`/`gfx1103` (RDNA3)
- `gfx1200`/`gfx1201` (RDNA4)
- Generic targets: `gfx10-3-generic`, `gfx11-generic`, `gfx12-generic`

**Instruction sets used:** XDL (matrix cores, gfx90a+), WMMA (gfx11+)

## Code Formatting

CK uses `clang-format`. Pre-commit hooks are available:
```bash
sudo projects/composablekernel/script/install_precommit.sh
```

## Current Branch Context

Branch `users/vpietila/ck/wan-fwd-conv-tuning` is focused on **grouped forward convolution performance tuning**. The modified `profiler/src/CMakeLists.txt` strips down ckProfiler to only compile grouped convolution ops and device instances, reducing build time during this tuning work.
