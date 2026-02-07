# Quick Start: Self-Contained Include Tests

## What This Does

Tests all 74 header files in `include/ck_tile/ops/gemm/` to ensure each one can be compiled independently.

## Building the Tests

```bash
# From repository root
cd build

# Configure with GPU target (required)
cmake -DGPU_TARGETS="gfx942" ..

# Build all include tests
ninja
```

## File Locations

```
example/ck_tile/03_gemm/
├── generate_include_tests.sh          # Script to regenerate tests
├── INCLUDE_TESTS.md                   # Full documentation
└── QUICK_START_INCLUDE_TESTS.md       # This file

test/ck_tile/gemm/
├── CMakeLists.txt                     # Builds all include tests
└── include_tests/                     # 74 test files
    ├── README.md                      # Documentation for tests
    ├── test_include_block_*.cpp       # Block-level tests (26)
    ├── test_include_kernel_*.cpp      # Kernel-level tests (12)
    ├── test_include_pipeline_*.cpp    # Pipeline tests (24)
    └── test_include_warp_*.cpp        # Warp-level tests (12)
```

## Regenerating Tests

If you add new headers to `include/ck_tile/ops/gemm/`:

```bash
cd example/ck_tile/03_gemm
./generate_include_tests.sh
```

This will regenerate all test files in `test/ck_tile/gemm/include_tests/`.

## Test Results

- **Compilation success** = Header is self-contained ✓
- **Compilation failure** = Header is missing includes ✗

## Example Usage

```bash
# Build specific test
ninja test_include_block_block_gemm_problem

# Run it (should exit with code 0)
./test/ck_tile/gemm/test_include_block_block_gemm_problem
echo $?  # Prints 0
```

## Coverage

| Category | Count | Example Headers |
|----------|-------|-----------------|
| Block implementations | 26 | `block_gemm_areg_breg_creg_v1.hpp` |
| Kernel templates | 12 | `gemm_kernel.hpp`, `streamk_gemm_kernel.hpp` |
| Pipeline implementations | 24 | `gemm_pipeline_ag_bg_cr_comp_v6.hpp` |
| Warp operations | 12 | `warp_gemm.hpp`, `warp_gemm_attribute_mfma.hpp` |
| **Total** | **74** | All headers in `include/ck_tile/ops/gemm/` |
