# Self-Contained Include Tests for CK_Tile GEMM

This document describes the self-contained include testing framework for all headers in `include/ck_tile/ops/gemm/`.

## Overview

A **self-contained include test** verifies that a header file can be compiled on its own without relying on other headers being included first. Each test includes exactly one header and attempts to compile - if successful, the header is self-contained.

## What Was Created

### 1. Test Generation Script
**Location**: `example/ck_tile/03_gemm/generate_include_tests.sh`

A bash script that:
- Finds all `.hpp` files in `include/ck_tile/ops/gemm/` recursively
- Generates a minimal `.cpp` test file for each header
- Places all test files in `test/ck_tile/gemm/include_tests/` directory

Usage:
```bash
cd example/ck_tile/03_gemm
./generate_include_tests.sh
```

### 2. Generated Test Files
**Location**: `test/ck_tile/gemm/include_tests/`

Contains **74 test files**, one for each header:
- `test_include_block_*.cpp` - Tests for block-level implementations
- `test_include_kernel_*.cpp` - Tests for kernel-level templates
- `test_include_pipeline_*.cpp` - Tests for pipeline implementations
- `test_include_warp_*.cpp` - Tests for warp-level operations

Each test file has this structure:
```cpp
// Copyright notice
// Test that <header_path> is self-contained
#include "ck_tile/ops/gemm/<path>/<header>.hpp"

int main()
{
    // Minimal test - if this compiles, the header is self-contained
    return 0;
}
```

### 3. CMake Integration
**Location**: `test/ck_tile/gemm/CMakeLists.txt`

Added CMake code that:
- Globs all test files from `include_tests/`
- Creates an executable target for each test file
- Applies necessary compile options (matching the project's build settings)
- Targets are named `test_include_*` (without the `tile_` prefix)

## Building the Tests

### Prerequisites
You must set `GPU_TARGETS` when configuring CMake (tests only build for supported GPUs):

```bash
cd build
cmake -DGPU_TARGETS=gfx942 -DCK_CXX_STANDARD=17 -G Ninja ..
```

Or use the dev script:
```bash
../script/cmake-ck-dev.sh .. gfx942 -DCK_CXX_STANDARD=17 -G Ninja
```

### Build All Include Tests

```bash
# Build all tests
ninja

# Or build specific test
ninja test_include_block_block_gemm_problem
```

### Run Tests

The executables don't perform runtime tests - compilation success IS the test. If they compile, the headers are self-contained.

```bash
# Run a test (should exit immediately with code 0)
./test/ck_tile/gemm/test_include_block_block_gemm_problem
echo $?  # Should print 0
```

## Coverage

The tests cover all headers in these subdirectories:

| Directory | Headers | Description |
|-----------|---------|-------------|
| `block/` | 26 | Block-level GEMM implementations and policies |
| `kernel/` | 12 | Kernel templates and tile partitioners |
| `pipeline/` | 24 | Pipeline implementations and schedulers |
| `warp/` | 12 | Warp-level GEMM operations and attributes |
| **Total** | **74** | |

## Maintenance

### Adding New Headers

When new headers are added to `include/ck_tile/ops/gemm/`:

1. Run the generation script from the example directory:
   ```bash
   cd example/ck_tile/03_gemm
   ./generate_include_tests.sh
   ```

2. The CMakeLists.txt automatically picks up new test files (uses `file(GLOB)`)

3. Rebuild to compile the new tests:
   ```bash
   cd build
   ninja
   ```

### Fixing Failing Tests

If a test fails to compile:

1. Identify the missing dependency from the compiler error
2. Add the necessary `#include` to the header file
3. Rebuild to verify the fix

Example:
```
test_include_block_something.cpp:5:10: error: 'Tensor' was not declared
```

Fix: Add `#include "ck_tile/core/tensor/tensor.hpp"` to the header.

## Benefits

This testing approach provides:

1. **Header Independence** - Each header can be used standalone
2. **Clear Dependencies** - Missing includes are caught immediately
3. **Maintainability** - Prevents fragile include order dependencies
4. **Documentation** - The includes in each header clearly show what it depends on
5. **Refactoring Safety** - Changes that break self-containment are detected immediately

## Implementation Notes

- Tests use the same compile options as regular tests (`-mllvm -enable-noalias-to-md-conversion=0`)
- FP8 support is conditionally enabled via `CK_USE_OCP_FP8`
- Tests are only built when `GPU_TARGETS` matches supported architectures (gfx90a, gfx94x, gfx95x, gfx11x, gfx12x)
- The generation script can be run multiple times safely (overwrites existing test files)
- Tests are located in `test/ck_tile/gemm/include_tests/` to keep them separate from functional tests
