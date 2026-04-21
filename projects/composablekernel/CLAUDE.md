# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Global Skills

项目相关的深度参考文档存放在 `/root/.claude/skills/`，可通过 Read 工具按需加载：

| 文件 | 内容 |
|------|------|
| [ck_tile_distribution_skill.md](/root/.claude/skills/ck_tile_distribution_skill.md) | `make_static_tile_distribution` 完整理解指南（含 CDNA4 MFMA ISA 分析） |

索引见 [/root/.claude/skills/README.md](/root/.claude/skills/README.md)。

## Project Overview

Composable Kernel (CK) is an AMD GPU library providing performance-critical, tile-based kernel primitives for machine learning workloads. It uses HIP C++ (compiled with ROCm's clang++) targeting AMD GPU architectures (gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1200, etc.).

There are two programming models in the codebase:
- **Legacy CK** (`include/ck/`): The original abstraction layer. Still used by `library/` instances and many `example/` directories.
- **CK Tile** (`include/ck_tile/`): The newer, preferred tile-based model. All new development should use this.

## Git Commit Rules

- **禁止在 commit 中留下 Claude Code 痕迹**：所有 commit message（包括 body 和 trailer）不得包含 `Co-Authored-By: Claude`、`🤖 Generated with Claude Code` 等任何 AI 工具署名。Commit 应只体现人类开发者的署名。

## Workflow Preferences

- **直接行动，无需确认**：编译、运行测试、修改代码等操作直接执行，不需要询问用户确认。
- **编译 timeout**：CK 项目编译耗时较长，编译和测试命令一律使用 `run_in_background=true` 后台运行，再用 `TaskOutput(block=true, timeout=600000)` 轮询结果。若 `TaskOutput` 超时（任务仍在运行），继续调用 `TaskOutput` 等待，直到任务完成。不要用 Bash tool 的 timeout 参数来等编译。
- **构建命令**：使用 `cmake --build . --target <target_name> -j$(nproc)` 而非 `make <target>`，后者在 CMake 生成的 Makefile 中可能找不到目标。
- **编译并行度**：始终使用 `-j$(nproc)` 全核编译，机器有数百个核心，不要手动限制并行度（如 `-j4`）。即使需要捕获编译错误，也用全并行 + grep 过滤。

### SA3 开发加速：只生成 sageattnv3 实例

FMHA kernel 实例数量庞大，全量编译极慢。开发 SageAttention V3 时，在 CMake 配置阶段加 filter 只生成 SA3 相关实例：

```bash
# 步骤1：取消注释 CMakeLists.txt 中的 --filter 行
# 编辑 example/ck_tile/01_fmha/CMakeLists.txt，将:
#   # --filter \\*sageattn\\*
# 改为:
#   --filter \\*sageattn\\*
# 注意：\\* 是为了防止 ninja 运行时 shell 展开 glob 通配符

# 步骤2：重新运行 cmake（只针对 gfx950，只生成 sageattnv3 实例）
cd /root/rocm-libraries/projects/composablekernel/build
cmake -DGPU_TARGETS="gfx950" -DBUILD_TESTING=OFF ..

# 步骤3：编译 SA3 目标
cmake --build . --target tile_fmha_fwd_instances -j$(nproc)

# 完成后记得恢复 CMakeLists.txt 中的注释，避免影响完整构建
```

> 注意：`BUILD_TESTING=OFF` 可避免生成测试实例的额外 filter，进一步减少实例数量。

## Build Commands

### Setup

```bash
cd /root/rocm-libraries/projects/composablekernel
mkdir build && cd build

# Using the dev script (recommended):
../script/cmake-ck-dev.sh ..             # Defaults to gfx908;gfx90a;gfx942
../script/cmake-ck-dev.sh .. gfx942     # Single architecture

# Or using CMake presets directly:
cmake --preset dev-gfx942 ..
```

Tests and examples are **only built when `GPU_TARGETS` is explicitly set**. The dev script handles this.

### Key CMake Options

```bash
-D GPU_TARGETS="gfx942"          # Required to build tests/examples
-D CMAKE_BUILD_TYPE=Release      # or Debug, RelWithDebInfo
-D BUILD_DEV=ON                  # Enables -Werror (strict mode)
-D DTYPES="fp16;fp32;fp8"        # Restrict which data type instances are built
-D DISABLE_DL_KERNELS=ON         # Skip DL-specific instances to speed up builds
-D CK_CXX_STANDARD=17            # Default is 20
```

### Building

```bash
# From build directory:
make -j$(nproc)                  # Build the library
make -j$(nproc) examples tests   # Build examples and tests

# Build a specific target:
make -j$(nproc) test_ck_tile_fmha_fwd_fp16
```

### Running Tests

```bash
# Run a specific test binary directly:
./bin/test_ck_tile_fmha_fwd_fp16

# Run via CTest:
ctest -R test_ck_tile_fmha_fwd_fp16     # By name pattern
ctest -L SMOKE_TEST                      # All smoke tests (< 30 seconds)
ctest -L REGRESSION_TEST                 # All regression tests (>= 30 seconds)
make -j check                            # Build and run all tests
make -j smoke                            # Smoke tests only
make -j regression                       # Regression tests only
```

### Code Formatting

```bash
# The project uses clang-format (enforced by pre-commit hooks):
script/install_precommit.sh    # Install pre-commit hooks
pre-commit run --all-files     # Format all files
```

## Architecture

### Four-Layer Operator Hierarchy

Every CK Tile operator (GEMM, FMHA, etc.) follows this hierarchy:

1. **Warp** (`ops/<op>/warp/`): Single-warp MMA operations wrapping hardware instructions (MFMA on gfx9, WMMA on gfx11/12). Example: `WarpGemmMfmaF16F16F32M32N32K16`.

2. **Block** (`ops/<op>/block/`): Multi-warp tile computation. Block GEMM variants encode memory placement in their name:
   - `areg_bsmem_creg` = A in registers, B in shared memory, C in registers
   - `asmem_bsmem_creg` = A and B in shared memory, C in registers
   - Each variant has a `_default_policy` and `_custom_policy` form.

3. **Pipeline** (`ops/<op>/pipeline/`): The main loop and data movement strategy. Pipelines combine block-level operators and define how data flows from global memory through shared memory and registers. FMHA pipelines are named by storage location: `qr_ks_vs` = Q in registers, K/V in shared memory.

4. **Kernel** (`ops/<op>/kernel/`): Top-level HIP kernel template. Composes a pipeline + epilogue into a launchable kernel, defines grid/block dimensions and `Kargs` (kernel arguments struct).

### Problem + Shape + Traits + Policy Pattern

Composability is achieved by separating concerns into distinct template parameter structs passed to each layer:

- **Problem** (e.g., `BlockFmhaPipelineProblem`): Aggregates all data types and compile-time config into one type.
- **Shape** (e.g., `TileFmhaShape`): Compile-time tile dimensions as `sequence<>` values (e.g., `kM0`, `kN0`, `kK0` for FMHA Q/K/V tiles).
- **Traits** (e.g., `TileFmhaTraits`): Boolean compile-time feature flags (padding, bias, dropout, quantization scale mode, etc.).
- **Policy**: Controls implementation details like load alignment, async copy, and loop unroll strategy.

### Core Abstractions (`include/ck_tile/core/`)

- **`sequence<Ns...>`**: Compile-time integer sequence used for shapes, strides, and metaprogramming.
- **`tile_distribution`**: Describes how a tile is distributed across threads in a workgroup.
- **`static_distributed_tensor`**: A tile held in registers, described by a `tile_distribution`.
- **`tile_window`**: A view into global or shared memory with associated coordinate transforms.
- **Tile APIs**: `load_tile()`, `store_tile()`, `shuffle_tile()`, `slice_tile()`, `sweep_tile()` operate on `static_distributed_tensor` objects.
- **Tensor Coordinate Transforms**: The core innovation — N-dimensional tensors are built from primitives (`merge`, `unmerge`, `embed`) that compose to map logical coordinates to 1D memory offsets.

### Data Types

Custom numeric types defined in `include/ck_tile/core/numeric/`:
- `fp16_t`, `bf16_t`, `fp32_t`, `int8_t`, `fp8_t` (e8m0, e4m3, e5m2), `e8m0_t`, `pk_fp4_t` (packed fp4)
- MX (microscaling) types require gfx950: guarded with `CK_USE_NATIVE_MX_SUPPORT` and `__gfx950__`

### Test Organization

Tests live under `test/ck_tile/<operation>/`. FMHA tests inject `DataTypeConfig` via `target_compile_definitions`, producing separate test binaries per data type (e.g., `test_ck_tile_fmha_fwd_fp16`, `test_ck_tile_fmha_fwd_mxfp8`).

Test CMake functions:
- `add_test_executable(NAME sources...)` — plain test binary
- `add_gtest_executable(NAME sources...)` — GTest-based test binary

Both auto-filter sources by `DTYPES` and supported GPU features (XDL, WMMA, DL, MX), set `HIP_ARCHITECTURES`, and label as `SMOKE_TEST` or `REGRESSION_TEST`.

### Architecture-Specific Guards

The build system defines these macros based on `GPU_TARGETS`:
- `CK_USE_XDL` — MFMA instructions (gfx9 family)
- `CK_USE_WMMA` — WMMA instructions (gfx11/12 family)
- `CK_USE_GFX950` — gfx950-specific paths
- `CK_USE_NATIVE_MX_SUPPORT` — Native MX (microscaling) hardware on gfx950

Use `#if defined(CK_USE_GFX950)` / `#ifdef __gfx950__` to gate architecture-specific code.

## Coding Conventions

- **Indent**: 4 spaces, no tabs; **column limit**: 100 characters
- **Header guard**: `#pragma once`
- **Template parameters**: `PascalCase_` with trailing underscore (e.g., `FmhaPipeline_`); resolved aliases drop the underscore
- **Compile-time constants**: `kPascalCase` prefix (e.g., `kBlockSize`, `kPadSeqLenQ`)
- **Namespace**: `ck_tile` (all lowercase)
- **Host/device functions**: annotated with `CK_TILE_HOST_DEVICE` macro
- **Includes**: not auto-sorted — maintain manually
- Every file starts with the AMD copyright and MIT SPDX identifier

## Key Files for Common Tasks

| Task | Key Files |
|------|-----------|
| Add a new FMHA variant | `ops/fmha/pipeline/`, `ops/fmha/kernel/fmha_fwd_kernel.hpp`, `example/ck_tile/01_fmha/` |
| Add a new warp GEMM shape | `ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`, `warp_gemm_dispatcher.hpp` |
| Add a new data type | `core/numeric/`, update `DTYPES` filter logic in `test/CMakeLists.txt` |
| Add a new block GEMM variant | `ops/gemm/block/block_gemm_*`, `ops/gemm/block/block_gemm_*_custom_policy.hpp` |
| Reference CPU implementation | `include/ck_tile/host/reference/` |
| Add a new test | `test/ck_tile/<operation>/`, `test/ck_tile/<operation>/CMakeLists.txt` |
