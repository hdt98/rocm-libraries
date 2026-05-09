# Plan: Eliminate MIOpenDriver's Use of Internal MIOpen Symbols

## Context

MIOpen compiles to a shared library whose public API is defined solely in `include/miopen/miopen.h`. The `MIOpenDriver` application (under `driver/`) is a test/benchmark tool that should ideally use only this public C API. However, the driver extensively includes internal MIOpen C++ headers from `src/include/miopen/`, creating a dependency on private symbols. This means MIOpenDriver cannot be built against an installed MIOpen — it must be built in-tree, and it couples tightly to internal implementation details.

This plan catalogs every internal-header inclusion, classifies the reason for each leak, and proposes fixes.

---

## Category 1: `miopen::deref()` — Casting opaque handles to C++ objects (MOST PERVASIVE)

**Files affected:** ~38 driver files, 307 total call sites

**What it does:** `miopen::deref(handle)` casts an opaque C handle (e.g., `miopenTensorDescriptor_t`) to its internal C++ object (e.g., `miopen::TensorDescriptor&`), exposing all internal methods.

**Why it's used:** The driver needs to query properties of descriptors (lengths, strides, layout, data type, element count, etc.) and there are gaps in the public C API for some of these queries.

**Specific sub-cases:**

### 1a. Tensor descriptor queries via `miopen::deref(tensorDesc)`
- `.GetLengths()`, `.GetStrides()`, `.GetLayout_t()`, `.GetType()`, `.GetNumDims()`, `.GetElementSize()`, `.GetLayoutEnum()`, `.GetVectorLength()`
- **Used in:** `conv_driver.hpp`, `tensor_driver.hpp`, `bn_driver.hpp`, `layernorm_driver.hpp`, `t5layernorm_driver.hpp`, `adam_driver.hpp`, `getitem_driver.hpp`, `multimarginloss_driver.hpp`, `softmarginloss_driver.hpp`, `rope_driver.hpp`, `reducecalculation_driver.hpp`, `reduceextreme_driver.hpp`, `kthvalue_driver.hpp`, `mloPoolingHost.hpp`, `mloSoftmaxHost.hpp`, `mloConvHost.hpp`, `mloGroupNormHost.hpp`, `mloPReLUHost.hpp`, `activ_driver.hpp`, `ctc_driver.hpp`, `softmax_driver.hpp`, `CBAInferFusion_driver.hpp`, `miopen_ConvBatchNormActivHost.hpp`, `rnn_seq_driver.hpp`, `rnn_driver.hpp`, `dropout_driver.hpp`, `gemm_driver.hpp`, `tensorop_driver.hpp`, `glu_driver.hpp`, `addlayernorm_driver.hpp`
- **Fix:** The public API already has `miopenGetTensorDescriptor()` and `miopenGetTensorDescriptorSize()` which return datatype, dims, and strides. Add a thin wrapper function in the driver that calls these public APIs and returns the needed info. For `GetElementSize()`, use `miopenGetTensorNumBytes() / sizeof(type)`. For `GetLayoutEnum()` / `GetVectorLength()`, these may need new public API functions or the driver can infer layout from strides.

### 1b. Convolution descriptor queries via `miopen::deref(convDesc)`
- `.GetConvStrides()`, `.GetConvPads()`, `.GetConvDilations()`, `.GetGroupCount()`, `.GetSpatialDimension()`
- **Used in:** `conv_driver.hpp` (~40 sites), `CBAInferFusion_driver.hpp` (~8 sites)
- **Fix:** Public API already provides `miopenGetConvolutionDescriptor()` which returns pads/strides/dilations, `miopenGetConvolutionGroupCount()`, and `miopenGetConvolutionSpatialDim()`. Replace `miopen::deref()` calls with these C API calls. The driver already knows these values since it set them — so alternatively, store the values locally when setting the descriptor.

### 1c. Pooling descriptor direct assignment via `miopen::deref(poolDesc)`
- `pool_driver.hpp:406` — directly constructs a `miopen::PoolingDescriptor` C++ object and assigns it
- **Fix:** Use `miopenSetNdPoolingDescriptor()` / `miopenSet2dPoolingDescriptor()` from the public API instead

### 1d. Dropout descriptor field access via `miopen::deref(dropoutDesc)`
- `.stateSizeInBytes`, `.seed`, `.dropout`, `.use_mask`
- **Used in:** `dropout_gpu_emulator.hpp` (6 sites)
- **Fix:** Public API has `miopenDropoutGetStatesSize()` and `miopenGetDropoutDescriptor()` which return the needed values. Use those.

---

## Category 2: `MIOPEN_THROW` / `MIOPEN_THROW_IF` / `MIOPEN_THROW_HIP_STATUS` — Internal exception macros

**Headers:** `miopen/errors.hpp`
**Files affected:** Nearly every driver file (~130+ call sites)

**Why it's used:** Convenient error reporting that throws `miopen::Exception`.

**Fix:** Replace with standard C++ exceptions (`std::runtime_error`, `std::invalid_argument`). Define a simple driver-local error macro:
```cpp
#define DRIVER_THROW(msg) throw std::runtime_error(msg)
```
Also replace `catch(const miopen::Exception& ex)` in `main.cpp` with `catch(const std::exception& ex)` (which is already there as a second catch).

---

## Category 3: `MIOPEN_DECLARE_ENV_VAR_*` / `MIOPEN_GET_ENV_*` — Internal env var macros

**Headers:** `miopen/env.hpp`
**Files affected:** `conv_driver.hpp`, `CBAInferFusion_driver.hpp`, `random.hpp`

**Env vars declared:**
- `MIOPEN_DRIVER_PAD_BUFFERS_2M`
- `MIOPEN_DRIVER_USE_GPU_REFERENCE`
- `MIOPEN_DRIVER_SUBNORM_PERCENTAGE`
- `MIOPEN_DRIVER_CONV_WORKSPACE_SIZE_ADJUST`
- `MIOPEN_DEBUG_DRIVER_PRNG_SEED`

**Fix:** Replace with `std::getenv()` and local parsing. These are driver-specific env vars anyway — they don't need the MIOpen macro infrastructure.

---

## Category 4: `miopen::debug::*` — Direct manipulation of internal debug globals

**Header:** `miopen/logger.hpp`, `miopen/execution_context.hpp`
**File:** `conv_driver.hpp` (lines 67-116)

**Globals accessed:**
- `miopen::debug::LoggingQuiet`
- `miopen::debug::FindEnforceDisable`
- `miopen::debug::IsWarmupOngoing`
- `miopen::debug::AlwaysEnableConvDirectNaive`

**Why it's used:** `AutoMiopenWarmupMode` and `AutoPrepareForGpuReference` suppress logging and force specific solver behavior during warmup/reference runs.

**Fix:** This is the hardest category. These debug knobs have no public API equivalent. Options:
1. **Add public API functions** like `miopenSetDebugFlags()` to control these behaviors
2. **Use environment variables** — MIOpen already reads `MIOPEN_DEBUG_*` env vars; the driver could set them via `setenv()` before calls
3. **Accept this as an in-tree-only feature** — warmup mode is a driver-specific optimization, not something external consumers need

**Recommendation:** Option 2 (use `setenv`/`unsetenv` for the debug env vars) is the least invasive. Option 1 is better long-term if these controls are valuable to other users.

---

## Category 5: `miopen::Handle` / `miopen::HipEventPtr` / `miopen::make_hip_event()` — Internal handle/event types

**Header:** `miopen/handle.hpp`
**Files:** `driver.hpp`, `timer.hpp`, `bn_driver.hpp`, `CBAInferFusion_driver.hpp`

**Why it's used:**
- `timer.hpp` uses `miopen::make_hip_event()` and `miopen::HipEventPtr` for GPU timing
- `driver.hpp` includes it for the `MIOPEN_BACKEND_*` preprocessor checks and handle access

**Fix:** Replace `miopen::make_hip_event()` / `miopen::HipEventPtr` with direct HIP API calls (`hipEventCreate`/`hipEventDestroy` with a custom RAII wrapper). The preprocessor backend check (`MIOPEN_BACKEND_HIP`) comes from `miopen/config.h` — see Category 8.

---

## Category 6: `MIOPEN_LOG_CUSTOM` / `miopen::LoggingLevel` — Internal logging

**Header:** `miopen/logger.hpp`
**Files:** `conv_driver.hpp`, `driver.hpp`, `driver.hpp`, `rnn_seq_driver.hpp`

**Fix:** Replace with `std::cerr`/`fprintf(stderr, ...)` or a simple driver-local logging utility. These are just debug print statements.

---

## Category 7: `miopen::TensorDescriptor` C++ class — Used as a value type

**Header:** `miopen/tensor.hpp`
**Files:** `tensor_driver.hpp`, `bn_driver.hpp`, `mloPoolingHost.hpp`, various `*_driver.hpp`

**Why it's used:** The `GpumemTensor` template in `driver.hpp` stores a `.desc` member of type `miopen::TensorDescriptor`. Batch norm driver calls `miopen::DeriveBNTensorDescriptor()` directly.

**Fix:** Replace internal `miopen::TensorDescriptor` usage with the opaque `miopenTensorDescriptor_t` handle throughout. For `DeriveBNTensorDescriptor`, the public API already has `miopenDeriveBNTensorDescriptor()` (used in `CBAInferFusion_driver.hpp:768`).

---

## Category 8: `miopen/config.h` — Build config macros

**Header:** `miopen/config.h` (generated at build time)
**Files:** `main.cpp`, `util_driver.hpp`, `gemm_driver.hpp`

**Used for:** `MIOPEN_BACKEND_OPENCL`, `MIOPEN_BACKEND_HIP`, `MIOPEN_USE_ROCBLAS`

**Fix:** This is quasi-public — it's needed to know which backend to compile for. Options:
1. Make `config.h` an officially public header (it's already installed)
2. Use CMake variables directly in the driver's build instead of reading config.h
3. Define `MIOPEN_BACKEND_HIP` etc. via CMake target properties

**Recommendation:** Option 2/3 — pass the backend define via CMake (`target_compile_definitions`). The driver's CMakeLists.txt already knows the backend.

---

## Category 9: Internal helper functions used for CPU reference

**Headers & functions:**
- `miopen/algorithm.hpp` — `miopen::sort()` etc.
- `miopen/conv_algo_name.hpp` — `miopen::ConvolutionAlgoToString()`
- `miopen/convolution.hpp` — `miopen::ConvolutionDescriptor` C++ type
- `miopen/find_controls.hpp` — find mode enums
- `miopen/conv/solvers.hpp` — solver names
- `miopen/tensor_extra.hpp` — `GetTensorStrides`, layout extras
- `miopen/tensor_layout.hpp` — `tensor_layout_get_default()`, `tensor_layout_to_strides()`
- `miopen/float_equal.hpp` — `miopen::float_equal()`
- `miopen/ford.hpp` — `miopen::ford()` parallel for loop
- `miopen/par_for.hpp` — `miopen::par_for()`
- `miopen/bfloat16.hpp` — `bfloat16` type
- `miopen/stringutils.hpp` — `miopen::StartsWith()`
- `miopen/demangle.hpp` — type name demangling
- `miopen/reduce_common.hpp` — type conversion helpers
- `miopen/tensor_view_utils.hpp` — tensor view construction
- `miopen/prelu/utils.hpp` — PReLU utilities
- `miopen/pooling.hpp` — `PoolingDescriptor` C++ type
- `miopen/dropout.hpp` — `DropoutDescriptor` C++ type
- `miopen/rnn.hpp` — `RNNDescriptor` C++ type
- `miopen/batch_norm.hpp` — batch norm descriptor type
- `miopen/tensor_ops.hpp` — tensor operations C++ API
- `miopen/gemm_v2.hpp` — `GemmDescriptor`, `CallGemm`

**Fix (per sub-group):**
- **Simple utilities** (`float_equal`, `ford`, `par_for`, `StartsWith`, `demangle`, `bfloat16`, `reduce_common`): Copy or reimplement these trivial utilities locally in the driver. They are small helper functions.
- **Tensor layout helpers** (`tensor_layout.hpp`, `tensor_extra.hpp`, `tensor_view_utils.hpp`): These are used in `InputFlags.cpp` and CPU reference code. Reimplement the needed subset locally or add equivalent public API functions.
- **Descriptor C++ types** (`convolution.hpp`, `pooling.hpp`, `dropout.hpp`, `rnn.hpp`, `batch_norm.hpp`): Only used for `miopen::deref()` access — eliminating deref (Category 1) eliminates these includes.
- **GEMM** (`gemm_v2.hpp`): `GemmDescriptor`, `CallGemm`, `CallGemmStridedBatched` — these are entirely internal. The GEMM driver should either use a public GEMM API (if one exists or is added) or be removed/moved into tests.
- **`ConvolutionAlgoToString()`**: Used only for printing. Implement a local string lookup table in the driver.
- **`miopen/conv/solvers.hpp`**: Check actual usage — may only be included transitively.

---

## Category 10: Hidden/Undocumented API functions

**File:** `conv_driver.hpp:48-49`
```cpp
extern "C" MIOPEN_EXPORT miopenStatus_t
miopenHiddenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc, int findMode);
```

**Comparison with public API:**
- **`miopenSetConvolutionFindMode()`** (`src/convolution_api.cpp:193`): Validates the `findMode` — checks it is within `[Begin_, End_)` and rejects `DeprecatedFastHybrid`. Returns `miopenStatusBadParm` on invalid input.
- **`miopenHiddenSetConvolutionFindMode()`** (`src/convolution_api.cpp:298`): Does **no validation** — blindly casts the `int` and sets it. Can accept invalid or deprecated values.

In the driver's usage (`conv_driver.hpp:845-848`), the same value (`miopenConvolutionFindModeNormal`) is passed to both calls, so they produce identical results. The hidden call on line 846-848 is redundant — the comment says "Repeat via hidden API" which suggests it was a smoke test of the hidden API itself, not a functional necessity.

**Fix:** Remove the hidden call (lines 46-49, 846-848). The public `miopenSetConvolutionFindMode()` on line 845 is sufficient and is strictly safer due to its validation.

---

## Priority Order for Implementation

1. **Category 10** (Hidden API) — trivial, 1 line removal
2. **Category 2** (MIOPEN_THROW) — mechanical replacement, high impact on include count
3. **Category 3** (ENV_VAR macros) — straightforward, 5 call sites
4. **Category 6** (MIOPEN_LOG) — straightforward, ~10 call sites
5. **Category 1** (miopen::deref) — largest effort, needs public API helper wrappers
6. **Category 5** (handle.hpp/HipEventPtr) — moderate, need RAII wrapper
7. **Category 8** (config.h) — CMake change
8. **Category 9** (helper functions) — most tedious, many small reimplementations
9. **Category 7** (TensorDescriptor as value type) — requires refactoring GpumemTensor
10. **Category 4** (debug globals) — needs design decision on approach

## Verification

After each category of changes:
1. Build MIOpenDriver: `cmake --build build --target MIOpenDriver`
2. Run a basic test: `./build/bin/MIOpenDriver conv -n 1 -c 1 -H 32 -W 32 -k 1 -y 3 -x 3 -V 1`
3. Verify no internal `#include <miopen/...>` remain (except `miopen/miopen.h` and possibly `miopen/config.h`)
4. Check that no `miopen::` namespace symbols are used (except through the public C API)
