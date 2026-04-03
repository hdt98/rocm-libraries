# rocm_ck Tests

## Testing Strategy

Tests are organized in two tiers, each with a different purpose.

### Tier 1: Host-side unit tests

Compiled with g++ (no HIP). Tests all schema types: Signature, resolve(), GemmSpec, Args, DataType, Layout.
Runs in seconds, no GPU needed. **This is where coverage lives.**

```bash
cd tests/
cmake -B build -S . -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure
```

### Tier 2: GPU runtime tests (examples)

Examples serve as integration tests: they launch kernels, compile device code, and verify numerical
correctness on real hardware. Device compilation validation (type maps, CK Tile template
instantiation) is covered here since device compilation is slow (10-30s per variant).
GPU tests are **not** part of this test suite. They live in `examples/`.

## Test Files

| File | Tier | What it tests |
|------|------|---------------|
| `test_datatype.cpp` | 1 | DataType enum, `dataTypeBits()`, `dataTypeName()` |
| `test_layout.cpp` | 1 | Layout enum, `layoutName()`, `isValidLayoutForRank()` |
| `test_signature.cpp` | 1 | Signature construction, Tensor/Scalar/Op fields |
| `test_resolve.cpp` | 1 | `resolve()` — dtype cascade, rank/layout propagation, FMHA pattern, scalars |
| `test_gemm_spec.cpp` | 1 | `isValidWaveTile()`, `makeSpec()`, physical tensor table, epilogue ops |
| `test_elementwise_spec.cpp` | 1 | ElementwiseSpec `makeSpec()` validation, alignment checks |
| `test_args.cpp` | 1 | Args ABI: sizes, offsets, alignment, trivially-copyable |
| `test_physical_tensor.cpp` | 1 | PhysicalTensor, TensorName NTTP string |
| `test_validate.cpp` | 1 | `validate()` — Args slot checking against spec's physical tensor table |
| `test_schema_compatibility.cpp` | 1 | ABI stability of Args, TensorArg, ScalarValue across versions |
| `test_compile_errors.cpp` | — | Reference document: expected-failure cases (active tests in `compile_fail/`) |

## Static Assert Policy

**Unit tests verify behavior; static_asserts protect users.**

Most static_asserts have been migrated out of headers into this test suite. Headers retain
only critical guardrails that produce clear compiler errors when a user misconfigures a kernel:

- ABI size/alignment assertions in `args.hpp` (prevent silent memory corruption)
- `isValidWaveTile()` check inside `makeSpec()` (catch invalid tile configs at compile time)
- Device-side assertions in `gemm_dev.hpp` (catch invalid GemmSpec at instantiation)

## Compile-Fail Tests

The `compile_fail/` directory contains negative tests that verify invalid configurations
fail at compile time with actionable error messages. Each `.cpp` file defines one invalid
spec; CMake marks these as `WILL_FAIL` tests. Coverage includes SSA violations, missing
dtype, invalid wave tiles, conflicting layouts, and other constraint violations.

See `test_compile_errors.cpp` for a reference catalog of expected-failure cases.

## Running Tests

```bash
# From the tests/ directory:
cd tests/
cmake -B build -S . -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure

# Run a specific test
ctest --test-dir build -R test_resolve --output-on-failure
```
