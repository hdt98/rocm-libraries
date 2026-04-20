# Wire GPU conv fwd into graph executor, add ShallowGpuTensor, expand test coverage

## Motivation

The GPU graph executor (`GpuReferenceGraphExecutor`) previously only supported a dummy pointwise plan. This PR wires the existing `GpuFpReferenceConvolution::fprop` implementation into the executor, enabling it to run convolution forward graphs end-to-end on device memory.

Both GPU and CPU executors now follow the same pattern: variant pack `void*` pointers are wrapped in shallow tensors (`ShallowGpuTensor` for GPU, `ShallowTensor` for CPU) that enforce memory boundaries — GPU rejects host access, CPU rejects device access — before being passed to the reference convolution via `TensorBase&`.

## Changes

### New files
- **`ShallowGpuTensor.hpp`** — non-owning device tensor mirroring `ShallowTensor` (host-only) from data_sdk
- **`ShallowDeviceOnlyMigratableMemory.hpp`** — device-only memory wrapper; throws on all host access
- **`GpuConvolutionFwdPlan.hpp`** — GPU conv fwd plan executor wrapping variant pack pointers in `ShallowGpuTensor`
- **`GpuConvolutionFwdSignatureKey.hpp`** — registry key + 5 registered type signatures
- **`ConvolutionFwdGraphTestUtils.hpp`** — shared graph builder for conv fwd tests (single-type and mixed-type overloads)
- **`TestShallowGpuTensor.cpp`** — 14 unit tests for `ShallowDeviceOnlyMigratableMemory` and `ShallowGpuTensor`
- **`TestGpuConvolutionFwdSignatureKey.cpp`** — signature key equality, hashing, and node construction tests
- **`TestGpuConvolutionFwdPlan.cpp`** — plan builder, plan execution vs CPU reference, and rejection tests

### Modified files
- **`GpuFpReferenceConvolution.hpp`** — extract raw-pointer private `fprop`; public overload calls `deviceData()` then delegates. `validateInput` now takes dims vectors. dgrad/wgrad updated to match.
- **`GpuReferenceGraphExecutor.hpp`** — register conv fwd in `buildSignatureKey`; fix virtual tensor to use `rawDeviceData()` instead of `rawHostData()`
- **`GpuPlanRegistrySignatureKey.hpp`** — add `GpuConvolutionFwdSignatureKey` to the variant
- **`GpuReferenceValidationFactory.hpp`** — add missing include
- **`ConvolutionValidation.hpp`** (test_sdk) — primary overload now takes `vector<int64_t>` dims; `TensorBase` overload delegates to it
- **`CMakeLists.txt`** — register new test sources and link `hipdnn_gpu_ref`

### Bug fixes
- **P0**: `TestGpuReferenceGraphExecutor` conv tests were passing host pointers to GPU kernels — fixed to use `hipMalloc`/`hipMemcpy`
- **P0**: Virtual tensor variant pack used `rawHostData()` but GPU plans expect device pointers — fixed to `rawDeviceData()`
- **P1**: Device memory leak on early `ASSERT_EQ` failure in test helpers — added cleanup lambda
- **P1**: Restored defensive `else if(nDims == 5)` with explicit throw on unsupported dimensions

## Registered type signatures

| # | X | W | Y | Compute |
|---|---|---|---|---------|
| 1 | FLOAT | FLOAT | FLOAT | FLOAT |
| 2 | HALF | HALF | HALF | FLOAT |
| 3 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT |
| 4 | HALF | HALF | FLOAT | FLOAT |
| 5 | BFLOAT16 | BFLOAT16 | FLOAT | FLOAT |

## Test plan

- [ ] `hipdnn_integration_tests_unit_tests --gtest_filter="TestGpuConvolutionFwdPlan*.*"` — 10 tests (3 CPU-only, 7 GPU)
- [ ] `hipdnn_integration_tests_unit_tests --gtest_filter="TestGpuReferenceGraphExecutor*.*"` — 10 tests (6 existing + 4 new conv fwd)
- [ ] `hipdnn_integration_tests_unit_tests --gtest_filter="TestGpuConvolutionFwdSignatureKey.*"` — 4 tests
- [ ] `hipdnn_gpu_ref_tests --gtest_filter="TestShallowDeviceOnlyMigratableMemory.*:TestShallowGpuTensor.*"` — 14 tests
- [ ] Existing pointwise tests unchanged (no regressions)
- [ ] Existing GPU ref conv tests pass: `hipdnn_gpu_ref_tests --gtest_filter="*GpuConvFwdRef*"`

### Test count summary

| Target | New tests |
|--------|-----------|
| `hipdnn_gpu_ref_tests` | 14 (ShallowGpuTensor + memory) |
| `hipdnn_integration_tests_unit_tests` | 21 (plan builder, execution, signature key, executor conv fwd) |
| **Total** | **35** |
