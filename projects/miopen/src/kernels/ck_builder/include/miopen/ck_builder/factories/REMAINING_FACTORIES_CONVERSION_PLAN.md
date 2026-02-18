# CK Builder: Remaining Factories Conversion Plan

## Context

The grouped 2D forward convolution has been fully ported to the CK Builder framework:
- 24 factory instance headers (`.hpp`) in `factories/grouped_conv_fwd/`
- 96 `.cpp` source files in `factories/grouped_conv2d_fwd/`
- 8 instance data helpers in `instance_data/`
- 1 factory dispatch header (`grouped_convolution_forward.hpp`)
- Static compile-time tests

This plan covers porting the **remaining** `DeviceOperationInstanceFactory` uses found in MIOpen
(documented in `DeviceOperationInstanceFactory_uses.md`) to the CK Builder framework.

## Documentation References

- [CK Builder README](../../README.md) - Overview of the CK Builder system
- [Instance Data README](../../instance_data/README.md) - Instance data struct documentation
- [Grouped Conv Fwd Header Conversion Plan](../../GROUPED_CONV_FWD_CONVERSION_PLAN.md) - Completed header conversion
- [Grouped Conv2D Fwd CPP Conversion Plan](../../../factories/grouped_conv2d_fwd/CPP_CONVERSION_PLAN.md) - Completed .cpp conversion
- [DeviceOperationInstanceFactory Uses](../../../../../DeviceOperationInstanceFactory_uses.md) - All factory uses in MIOpen

## Existing Infrastructure (Reusable)

The following files are already implemented and should be reused where applicable:

### Common Structures (`instance_data/common.hpp`)
- `ThreadBlock` - Block configuration (block_size, tile_size.m/n/k)
- `TransferABC` - Data transfer configurations for A, B, C tensors
- `ElementwiseOps` - Input/weight/output elementwise operations
- `TensorDescriptor` - Layout + data type + compute type
- `ConvSignature` - Already supports all directions via `ckb::ConvDirection`:
  - `FORWARD`, `BACKWARD_DATA`, `BACKWARD_WEIGHT`

### Kernel Instantiation (`kernel_instantiation.hpp`)
- `add_device_operation_instances<auto arr>(kernels)` - Main entry point
- `concat()` / `concat2()` - Compile-time array merging
- `filter_array_by_modulo()` - Sharding support
- `instantiate_kernel<KernelDescriptor>(kernels)` - Single kernel instantiation

### Factory Base (`factories/device_operation_instance_factory.hpp`)
- All data type aliases (F16, BF16, F32, TF32, I8, F8, BF8, etc.)
- All layout aliases (GNHWC, NHWGC, GKYXC, GNDHWC, NDHWGC, etc.)
- All elementwise operation aliases (PassThrough, Scale, Bilinear, Clamp, etc.)
- `DeviceOperationInstanceFactory<DeviceOp>` primary template declaration

---

## Factories to Port

From `DeviceOperationInstanceFactory_uses.md`, the following factories remain:

| # | MIOpen File | Factory Alias | Base Class | Dims | Direction |
|---|------------|---------------|------------|------|-----------|
| 1 | `implicitgemm_ck_util.hpp` | `DeviceOpGWrwPtrs` | `DeviceGroupedConvBwdWeight` | 2D | WrW |
| 2 | `implicitgemm_ck_util.hpp` | `DeviceOpGBwdPtrs` | `DeviceGroupedConvBwdDataMultipleD` | 2D | Bwd |
| 3 | `implicitgemm_ck_util.hpp` | `DeviceOpGBwdWeightDefaultPtrs` | `DeviceGroupedConvBwdWeight` | 3D | WrW |
| 4 | `implicitgemm_ck_util.hpp` | `DeviceOpGBwdWeightBilinearPtrs` | `DeviceGroupedConvBwdWeightMultipleD` | 3D | WrW |
| 5 | `implicitgemm_ck_util.hpp` | `DeviceOpGBwdWeightScalePtrs` | `DeviceGroupedConvBwdWeightMultipleD` | 3D | WrW |
| 6 | `conv_hip_implicit_gemm_fwd_xdlops.cpp` | `DeviceOpPtrs` | `DeviceConvFwd` | 2D | Fwd |
| 7 | `conv_hip_implicit_gemm_bwd_data_xdlops.cpp` | `DeviceOpBwdPtrs` | `DeviceConvBwdData` | 2D | Bwd |
| 8 | `conv_hip_implicit_gemm_grouped_wrw_xdlops.cpp` | `DeviceOpGWrwPtrs` | `DeviceGroupedConvBwdWeight` | 2D | WrW |
| 9 | `conv_hip_implicit_gemm_3d_grouped_fwd_xdlops.cpp` | `DeviceOpGFwdBilinearPtrs` | `DeviceGroupedConvFwdMultipleABD` | 3D | Fwd |
| 10 | `conv_hip_implicit_gemm_3d_grouped_fwd_xdlops.cpp` | `DeviceOpGFwdScalePtrs` | `DeviceGroupedConvFwdMultipleABD` | 3D | Fwd |
| 11 | `conv_hip_implicit_gemm_3d_grouped_fwd_xdlops.cpp` | `DeviceOpGFwdDefaultPtrs` | `DeviceGroupedConvFwdMultipleABD` | 3D | Fwd |
| 12 | `conv_hip_implicit_gemm_3d_grouped_bwd_xdlops.cpp` | `DeviceOpGBwdBilinearPtrs` | `DeviceGroupedConvBwdDataMultipleD` | 3D | Bwd |
| 13 | `conv_hip_implicit_gemm_3d_grouped_bwd_xdlops.cpp` | `DeviceOpGBwdScalePtrs` | `DeviceGroupedConvBwdDataMultipleD` | 3D | Bwd |
| 14 | `conv_hip_implicit_gemm_3d_grouped_bwd_xdlops.cpp` | `DeviceOpGBwdDefaultPtrs` | `DeviceGroupedConvBwdDataMultipleD` | 3D | Bwd |
| 15 | `conv_ck_igemm_fwd_bias_res_add_activ_fused.cpp` | `DeviceOp` | `DeviceGroupedConvFwdMultipleABD` | 3D | Fwd (fused) |
| 16 | `conv_ck_igemm_grp_fwd_activ_fused.cpp` | `DeviceOpGFwdActPtrs` | `DeviceGroupedConvFwdMultipleABD` | N-D | Fwd (fused) |
| 17 | `conv_ck_igemm_grp_fwd_bias_activ_fused.cpp` | `DeviceOpGFwdBiasActivPtrs` | `DeviceGroupedConvFwdMultipleABD` | N-D | Fwd (fused) |

---

## Phase Overview

| Phase | Description | New Instance Data | Factory Headers | .cpp Files | Dispatch Headers |
|-------|-------------|-------------------|-----------------|------------|------------------|
| 1 | Grouped Conv 3D/1D Forward (remaining .cpp) | 0 | 0 | ~78 | 0 |
| 2 | Grouped Conv Backward Data (2D+3D) | 3 | ~9 | ~91 | 3 |
| 3 | Grouped Conv Backward Weight (2D+3D) | 4 | ~12 | ~141 | 3 |
| 4 | Forward Fused Operations | 0 | ~8 | ~50+ | 5 |
| 5 | Non-Grouped Convolutions | 2 | ~2 | ~10 | 2 |
| **Total** | | **~9** | **~31** | **~370+** | **~13** |

---

## Phase 1: Grouped Conv 3D/1D Forward (Remaining .cpp Files)

**Covers entries:** #9-11 (3D forward default/bilinear/scale)

The forward factory instance headers (`.hpp`) are already converted and shared across 1D/2D/3D.
The factory dispatch header (`grouped_convolution_forward.hpp`) already dispatches for 1D and 3D.
Only the `.cpp` source files need conversion.

### 1.1 CK Source Locations

| Directory | Files | Description |
|-----------|-------|-------------|
| `grouped_conv3d_fwd/xdl/` | ~50 | XDL 3D forward |
| `grouped_conv3d_fwd/wmma/` | ~24 | WMMA 3D forward |
| `grouped_conv1d_fwd/xdl/` | 4 | XDL 1D forward |
| **Total** | **~78** | |

### 1.2 Work Items

#### Step 1.2.1: Convert .cpp Source Files

**Source:** `composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv{1,3}d_fwd/`
**Target:** `factories/grouped_conv{1,3}d_fwd/`

Follow the exact same conversion pattern as the completed 2D forward conversion
(`CPP_CONVERSION_PLAN.md`). The factory functions used are the same ones already in
`factories/grouped_conv_fwd/device_grouped_conv_fwd_*_instance.hpp`.

**Key difference from 2D:** 3D files use 3D layouts (GNDHWC, GKZYXC, GNDHWK, NDHWGC, NDHWGK, etc.)
and pass `spatial_dim = 3`. 1D files use 1D layouts (GNWC, GKXC, GNWK) and pass `spatial_dim = 1`.

#### Step 1.2.2: Update CMakeLists.txt

```cmake
# Grouped Conv3D Forward
set(GROUPED_CONV3D_FWD_XDL_SOURCES ...)
set(GROUPED_CONV3D_FWD_WMMA_SOURCES ...)

# Grouped Conv1D Forward
set(GROUPED_CONV1D_FWD_XDL_SOURCES ...)
```

### 1.3 Checklist

| Item | Status |
|------|--------|
| .cpp files: `grouped_conv3d_fwd/` (~74 files) | [ ] |
| .cpp files: `grouped_conv1d_fwd/` (4 files) | [ ] |
| CMakeLists.txt updated | [ ] |

---

## Phase 2: Grouped Conv Backward Data

**Covers entries:** #2 (2D), #12-14 (3D default/bilinear/scale)

### 2.1 CK Template Classes

| CK Template Class | File | Parameters |
|---|---|---|
| `DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1` | `device_grouped_conv_bwd_data_xdl_instance.hpp` | ~44 positional params (similar to fwd XDL) |
| `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle` | `device_grouped_conv_bwd_data_wmma_f16_instance.hpp` | WMMA variant |
| `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3` | `device_grouped_conv_bwd_data_wmma_v3_instances.hpp` | V3 variant |
| `DeviceGroupedConvBwdData_Xdl_CShuffle_Transpose` | `device_grouped_conv_bwd_data_transpose_xdl_instance.hpp` | Transpose variant |

### 2.2 CK Source Locations

**Instance Headers:** `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/` (9 files)

**Source Files:**

| Directory | Files | Description |
|-----------|-------|-------------|
| `grouped_conv2d_bwd_data/xdl/` | 27 | XDL 2D backward data |
| `grouped_conv2d_bwd_data/wmma/` | 12 | WMMA 2D backward data |
| `grouped_conv3d_bwd_data/xdl/` | 28 | XDL 3D backward data |
| `grouped_conv3d_bwd_data/wmma/` | 12 | WMMA 3D backward data |
| `grouped_conv3d_bwd_data_bilinear/xdl/` | 4 | XDL 3D bilinear |
| `grouped_conv3d_bwd_data_bilinear/wmma/` | 2 | WMMA 3D bilinear |
| `grouped_conv3d_bwd_data_scale/xdl/` | 4 | XDL 3D scale |
| `grouped_conv3d_bwd_data_scale/wmma/` | 2 | WMMA 3D scale |
| **Total** | **91** | |

### 2.3 Work Items

#### Step 2.3.1: Create Instance Data Helpers

Create new helper files in `instance_data/`:

| File | Helper Function | Returns | Based On |
|------|----------------|---------|----------|
| `xdl_bwd_data.hpp` | `DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NumDTensor>()` | `XdlBwdDataInstance` | `xdl.hpp` (similar params) |
| `wmma_bwd_data.hpp` | `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<NumDTensor>()` | `WmmaBwdDataInstance` | `wmma.hpp` (similar params) |
| `wmma_v3_bwd_data.hpp` | `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3<NumDTensor>()` | Stub (unsupported) | `wmma_v3.hpp` pattern |

**Key differences from forward XDL helper:**
- `ConvSignature.direction` = `ckb::ConvDirection::BACKWARD_DATA`
- `ConvSpecialization` uses backward data values (DEFAULT, FILTER_1X1_STRIDE_1_PAD_0)
- Layout parameter semantics differ: "A" is output gradient, "B" is weight, "E" is input gradient
- Same `TransferABC` and `ThreadBlock` structures can be reused from `common.hpp`

**Approach:** Read the forward `xdl.hpp` helper function signature and the CK `device_grouped_conv_bwd_data_xdl_instance.hpp` template parameters side-by-side. Map each CK template parameter to a helper function parameter, following the same pattern used for forward. The algorithm struct (`XdlBwdDataAlgorithm`) will be structurally identical to `XdlAlgorithm` but with backward-data-specific specialization enum values.

#### Step 2.3.2: Convert Factory Instance Headers

**Source:** `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/`
**Target:** `factories/grouped_conv_bwd_data/`

| Source File | Target File | Helper Used |
|-------------|-------------|-------------|
| `device_grouped_conv_bwd_data_xdl_instance.hpp` | `device_grouped_conv_bwd_data_xdl_instance.hpp` | `xdl_bwd_data.hpp` |
| `device_grouped_conv_bwd_data_xdl_bilinear_instance.hpp` | `device_grouped_conv_bwd_data_xdl_bilinear_instance.hpp` | `xdl_bwd_data.hpp` |
| `device_grouped_conv_bwd_data_xdl_scale_instance.hpp` | `device_grouped_conv_bwd_data_xdl_scale_instance.hpp` | `xdl_bwd_data.hpp` |
| `device_grouped_conv_bwd_data_transpose_xdl_instance.hpp` | `device_grouped_conv_bwd_data_transpose_xdl_instance.hpp` | New transpose helper or extension |
| `device_grouped_conv_bwd_data_wmma_f16_instance.hpp` | `device_grouped_conv_bwd_data_wmma_f16_instance.hpp` | `wmma_bwd_data.hpp` |
| `device_grouped_conv_bwd_data_wmma_i8_instance.hpp` | `device_grouped_conv_bwd_data_wmma_i8_instance.hpp` | `wmma_bwd_data.hpp` |
| `device_grouped_conv_bwd_data_wmma_v3_instances.hpp` | `device_grouped_conv_bwd_data_wmma_v3_instances.hpp` | Stub |
| `device_grouped_conv_bwd_data_wmma_v3_bilinear_instance.hpp` | `device_grouped_conv_bwd_data_wmma_v3_bilinear_instance.hpp` | Stub |
| `device_grouped_conv_bwd_data_wmma_v3_scale_instance.hpp` | `device_grouped_conv_bwd_data_wmma_v3_scale_instance.hpp` | Stub |

Follow the same two-stage conversion pattern as `GROUPED_CONV_FWD_CONVERSION_PLAN.md`:
- **Stage 1:** Create file structure with stub functions returning empty arrays
- **Stage 2:** Fill in converted template instantiations

#### Step 2.3.3: Create Factory Dispatch Headers

Create dispatch headers in `factories/`:

| File | Specializes On | CK Equivalent |
|------|---------------|---------------|
| `grouped_convolution_backward_data.hpp` | `DeviceGroupedConvBwdDataMultipleD<..., PassThrough, PassThrough, PassThrough>` | `grouped_convolution_backward_data.hpp` |
| `grouped_convolution_backward_data_bilinear.hpp` | `DeviceGroupedConvBwdDataMultipleD<..., PassThrough, PassThrough, Bilinear>` | `grouped_convolution_backward_data_bilinear.hpp` |
| `grouped_convolution_backward_data_scale.hpp` | `DeviceGroupedConvBwdDataMultipleD<..., PassThrough, PassThrough, Scale>` | `grouped_convolution_backward_data_scale.hpp` |

Each dispatch header follows the pattern of `grouped_convolution_forward.hpp`:
1. Include `.inc` files for each backend (XDL, WMMA) behind `#ifdef` guards
2. Specialize `DeviceOperationInstanceFactory<DeviceOp>` with a `GetInstances()` method
3. Dispatch via `if constexpr` on `NumDimSpatial` (1/2/3), layout, and data type
4. Call the appropriate `add_device_grouped_conv{2,3}d_bwd_data_*_instances()` functions

**Create `.inc` files** that declare the `add_device_*` functions. Follow the pattern of existing `.inc` files:
- `grouped_convolution_backward_data_xdl.inc`
- `grouped_convolution_backward_data_wmma.inc`
- `grouped_convolution_backward_data_wmma_cshufflev3.inc`
- (etc., as needed per backend variant)

#### Step 2.3.4: Convert .cpp Source Files

**Source:** `composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv{2,3}d_bwd_data*/`
**Target:** `factories/grouped_conv{2,3}d_bwd_data*/`

Follow the pattern from `CPP_CONVERSION_PLAN.md`:
1. Update includes to MIOpen headers
2. Update namespace from `ck::tensor_operation::device::instance` to `miopen::conv::ck_builder::instance`
3. Simplify function signature (remove template params from `DeviceGroupedConvBwdDataMultipleD`)
4. Add `namespace ckb = ck_tile::builder;` and `using namespace factories::grouped_conv_bwd_data;`
5. Define layout/specialization aliases
6. Convert `add_device_operation_instances` calls to template parameter syntax

**Parameter mapping for backward data (XDL/WMMA):**

| CK Template Parameter | Factory Function Parameter | Notes |
|----------------------|---------------------------|-------|
| `NDimSpatial` | `spatial_dim` | First argument |
| `ALayout` (e.g., `NHWGK`) | `input_layout` | Output gradient layout |
| `BLayout` (e.g., `GKYXC`) | `weight_layout` | Weight layout |
| `DsLayout` (e.g., `Empty_Tuple`) | `ds_layouts` | `{}` for empty |
| `ELayout` (e.g., `NHWGC`) | `output_layout` | Input gradient layout |
| `ConvSpec` | `conv_spec` | `ConvBwdDataDefault` etc. |
| `DsDataTypes` | `ds_types` | Optional, defaults to `{}` |
| `CDEElementOp` | `output_op` | Optional, defaults to `PASS_THROUGH` |

#### Step 2.3.5: Update CMakeLists.txt

Add source lists and file entries to `src/kernels/ck_builder/CMakeLists.txt`:

```cmake
# Grouped Conv2D Backward Data
set(GROUPED_CONV2D_BWD_DATA_XDL_SOURCES ...)
set(GROUPED_CONV2D_BWD_DATA_WMMA_SOURCES ...)

# Grouped Conv3D Backward Data
set(GROUPED_CONV3D_BWD_DATA_XDL_SOURCES ...)
set(GROUPED_CONV3D_BWD_DATA_WMMA_SOURCES ...)
set(GROUPED_CONV3D_BWD_DATA_BILINEAR_SOURCES ...)
set(GROUPED_CONV3D_BWD_DATA_SCALE_SOURCES ...)
```

#### Step 2.3.6: Add Static Tests

Create static tests in `static_tests/`:
- `grouped_conv_2d_bwd_data_xdl_f32.cpp`
- `grouped_conv_2d_bwd_data_wmma_f32.cpp`

#### Step 2.3.7: Update MIOpen Solver Files

Update the solver files to use the ck_builder factories behind `#ifdef CK_EXPERIMENTAL_BUILDER`:

- `src/include/miopen/solver/implicitgemm_ck_util.hpp` (DeviceOpGBwdPtrs)
- `src/solver/conv/conv_hip_implicit_gemm_3d_grouped_bwd_xdlops.cpp` (3D backward data)

### 2.4 Checklist

| Item | Status |
|------|--------|
| Instance data: `xdl_bwd_data.hpp` | [ ] |
| Instance data: `wmma_bwd_data.hpp` | [ ] |
| Instance data: `wmma_v3_bwd_data.hpp` (stub) | [ ] |
| Factory headers: 9 files in `factories/grouped_conv_bwd_data/` | [ ] |
| Dispatch header: `grouped_convolution_backward_data.hpp` | [ ] |
| Dispatch header: `grouped_convolution_backward_data_bilinear.hpp` | [ ] |
| Dispatch header: `grouped_convolution_backward_data_scale.hpp` | [ ] |
| `.inc` files for dispatch | [ ] |
| .cpp files: `grouped_conv2d_bwd_data/` (39 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_data/` (40 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_data_bilinear/` (6 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_data_scale/` (6 files) | [ ] |
| CMakeLists.txt updated | [ ] |
| Static tests | [ ] |
| Solver file integration | [ ] |

---

## Phase 3: Grouped Conv Backward Weight

**Covers entries:** #1, #3-5, #8

### 3.1 CK Template Classes

| CK Template Class | File | Notes |
|---|---|---|
| `DeviceGroupedConvBwdWeight_Xdl_CShuffle` | `device_grouped_conv_bwd_weight_xdl_instance.hpp` | Main XDL WrW, ~47 params. Has K0Per+K1 split (unlike forward's KPer). Optional compute types and transpose params. |
| `DeviceGroupedConvBwdWeight_Dl` | `device_grouped_conv_bwd_weight_dl_instance.hpp` | DL WrW, ~42 params. Different transfer structure than forward DL. |
| `DeviceGroupedConvBwdWeight_Wmma_CShuffle` | `device_grouped_conv_bwd_weight_wmma_*_instance.hpp` | WMMA WrW |
| `DeviceGroupedConvBwdWeight_TwoStage_Xdl` | `device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp` | Two-stage pipeline variant |
| `DeviceGroupedConvBwdWeight_TwoStage_Wmma` | `device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp` | Two-stage WMMA variant |
| `DeviceGroupedConvBwdWeight_V3_Xdl` | `device_grouped_conv_bwd_weight_v3_xdl_instance.hpp` | V3 variant |
| `DeviceGroupedConvBwdWeight_V3_Wmma` | `device_grouped_conv_bwd_weight_v3_wmma_instance.hpp` | V3 variant |

> **Note:** The backward weight XDL template has a **different parameter structure** than forward:
> - Uses `K0Per` + `K1` instead of forward's single `KPer`
> - No `DsLayout`/`DsDataTypes` (single output E, not multiple D tensors) for the base `DeviceGroupedConvBwdWeight`
> - The `MultipleD` variant (`DeviceGroupedConvBwdWeightMultipleD`) supports D tensors for bilinear/scale
> - Optional trailing parameters for compute types and transpose transfer dimensions

### 3.2 CK Source Locations

**Instance Headers:** `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/` (12 files)

**Source Files:**

| Directory | Files | Description |
|-----------|-------|-------------|
| `grouped_conv2d_bwd_weight/xdl/` | ~53 | XDL 2D (with gnhwc, ngchw, nhwgc layout variants + pipeline variants) |
| `grouped_conv2d_bwd_weight/dl/` | 6 | DL 2D |
| `grouped_conv2d_bwd_weight/wmma/` | ~7 | WMMA 2D |
| `grouped_conv3d_bwd_weight/xdl/` | ~48 | XDL 3D (with gndhwc, ngcdhw, ndhwgc layout variants) |
| `grouped_conv3d_bwd_weight/dl/` | 6 | DL 3D |
| `grouped_conv3d_bwd_weight/wmma/` | ~4 | WMMA 3D |
| `grouped_conv3d_bwd_weight_bilinear/xdl/` | 5 | XDL 3D bilinear |
| `grouped_conv3d_bwd_weight_bilinear/wmma/` | 2 | WMMA 3D bilinear |
| `grouped_conv3d_bwd_weight_scale/xdl/` | 5 | XDL 3D scale |
| `grouped_conv3d_bwd_weight_scale/wmma/` | 2 | WMMA 3D scale |
| **Total** | **~138** | |

### 3.3 Work Items

#### Step 3.3.1: Create Instance Data Helpers

Create new helper files in `instance_data/`:

| File | Helper Function | Returns | Notes |
|------|----------------|---------|-------|
| `xdl_bwd_weight.hpp` | `DeviceGroupedConvBwdWeight_Xdl_CShuffle()` | `XdlBwdWeightInstance` | New algorithm struct needed (K0Per+K1 split, optional params) |
| `xdl_two_stage_bwd_weight.hpp` | `DeviceGroupedConvBwdWeight_TwoStage_Xdl()` | `XdlTwoStageBwdWeightInstance` | Two-stage pipeline variant |
| `wmma_bwd_weight.hpp` | `DeviceGroupedConvBwdWeight_Wmma_CShuffle()` | `WmmaBwdWeightInstance` | |
| `dl_bwd_weight.hpp` | `DeviceGroupedConvBwdWeight_Dl()` | `DlBwdWeightInstance` | Different transfer structure from forward DL |

**Key differences from forward helpers:**
- `ConvSignature.direction` = `ckb::ConvDirection::BACKWARD_WEIGHT`
- `XdlBwdWeightAlgorithm` needs a different GEMM config struct with K0Per + K1 (instead of single KPer)
- No DsLayout/DsDataTypes for base version (the `MultipleD` variant adds these for bilinear/scale)
- Optional compute type and transpose transfer parameters at end of parameter list
- The `TransferABC` struct from `common.hpp` may need extension for the transpose transfer fields, or a new `TransferABC_BwdWeight` struct can be created

#### Step 3.3.2: Convert Factory Instance Headers

**Source:** `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/`
**Target:** `factories/grouped_conv_bwd_weight/`

Convert all 12 instance header files following the two-stage pattern.

#### Step 3.3.3: Create Factory Dispatch Headers

| File | Specializes On | CK Equivalent |
|------|---------------|---------------|
| `grouped_convolution_backward_weight.hpp` | `DeviceGroupedConvBwdWeight<..., PassThrough, PassThrough, PassThrough>` | `grouped_convolution_backward_weight.hpp` |
| `grouped_convolution_backward_weight_bilinear.hpp` | `DeviceGroupedConvBwdWeightMultipleD<..., PassThrough, Bilinear, PassThrough>` | `grouped_convolution_backward_weight_bilinear.hpp` |
| `grouped_convolution_backward_weight_scale.hpp` | `DeviceGroupedConvBwdWeightMultipleD<..., PassThrough, Scale, PassThrough>` | `grouped_convolution_backward_weight_scale.hpp` |

#### Step 3.3.4: Convert .cpp Source Files

Follow the same approach as Phase 2 but with backward weight parameter mapping:

**Parameter mapping for backward weight (XDL):**

| CK Template Parameter | Factory Function Parameter | Notes |
|----------------------|---------------------------|-------|
| `NDimSpatial` | `spatial_dim` | First argument |
| `ALayout` (e.g., `NHWGC`) | `input_layout` | Input activation layout |
| `BLayout` (e.g., `GKYXC`) | `weight_layout` | Output gradient → weight gradient layout |
| `ELayout` (e.g., `NHWGK`) | `output_layout` | Output gradient layout |
| `ConvSpec` | `conv_spec` | `ConvBwdWeightDefault` etc. |

> **Note:** The backward weight `DeviceGroupedConvBwdWeight` does NOT have `DsLayout`/`DsDataTypes` in
> its base form. The `DeviceGroupedConvBwdWeightMultipleD` variant (used for bilinear/scale) does.
> This means the factory functions for the base backward weight will have a different signature than
> forward/backward-data factories.

#### Step 3.3.5: Update CMakeLists.txt

Add source lists:

```cmake
# Grouped Conv2D Backward Weight
set(GROUPED_CONV2D_BWD_WEIGHT_XDL_SOURCES ...)
set(GROUPED_CONV2D_BWD_WEIGHT_DL_SOURCES ...)
set(GROUPED_CONV2D_BWD_WEIGHT_WMMA_SOURCES ...)

# Grouped Conv3D Backward Weight
set(GROUPED_CONV3D_BWD_WEIGHT_XDL_SOURCES ...)
set(GROUPED_CONV3D_BWD_WEIGHT_DL_SOURCES ...)
set(GROUPED_CONV3D_BWD_WEIGHT_WMMA_SOURCES ...)
set(GROUPED_CONV3D_BWD_WEIGHT_BILINEAR_SOURCES ...)
set(GROUPED_CONV3D_BWD_WEIGHT_SCALE_SOURCES ...)
```

### 3.4 Checklist

| Item | Status |
|------|--------|
| Instance data: `xdl_bwd_weight.hpp` | [ ] |
| Instance data: `xdl_two_stage_bwd_weight.hpp` | [ ] |
| Instance data: `wmma_bwd_weight.hpp` | [ ] |
| Instance data: `dl_bwd_weight.hpp` | [ ] |
| Factory headers: 12 files in `factories/grouped_conv_bwd_weight/` | [ ] |
| Dispatch header: `grouped_convolution_backward_weight.hpp` | [ ] |
| Dispatch header: `grouped_convolution_backward_weight_bilinear.hpp` | [ ] |
| Dispatch header: `grouped_convolution_backward_weight_scale.hpp` | [ ] |
| `.inc` files for dispatch | [ ] |
| .cpp files: `grouped_conv2d_bwd_weight/` (66 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_weight/` (61 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_weight_bilinear/` (7 files) | [ ] |
| .cpp files: `grouped_conv3d_bwd_weight_scale/` (7 files) | [ ] |
| CMakeLists.txt updated | [ ] |
| Static tests | [ ] |
| Solver file integration | [ ] |

---

## Phase 4: Forward Fused Variant Factories

**Covers entries:** #9-11 (bilinear/scale), #15 (scaleadd_scaleadd_relu), #16 (clamp), #17 (bias_clamp)

These use `DeviceGroupedConvFwdMultipleABD` with non-PassThrough elementwise operations.
The forward factory instance headers already include the converted template aliases for these variants.
What's needed are:
1. **Separate factory dispatch headers** for each elementwise operation combination
2. **`.inc` files** declaring the `add_device_*` functions
3. **`.cpp` source files** from the corresponding CK directories

### 4.1 Factory Dispatch Headers Needed

| File | Element Ops | CK Factory | MIOpen Entry |
|------|------------|-------------|-------------|
| `grouped_convolution_forward_bilinear.hpp` | PassThrough, PassThrough, Bilinear | `grouped_convolution_forward_bilinear.hpp` | #9 |
| `grouped_convolution_forward_scale.hpp` | PassThrough, PassThrough, Scale | `grouped_convolution_forward_scale.hpp` | #10 |
| `grouped_convolution_forward_scaleadd_scaleadd_relu.hpp` | PassThrough, PassThrough, ScaleAddScaleAddRelu | `grouped_convolution_forward_scaleadd_scaleadd_relu.hpp` | #15 |
| `grouped_convolution_forward_clamp.hpp` | PassThrough, PassThrough, Clamp | `grouped_convolution_forward_clamp.hpp` | #16 |
| `grouped_convolution_forward_bias_clamp.hpp` | PassThrough, PassThrough, AddClamp | `grouped_convolution_forward_bias_clamp.hpp` | #17 |

Each dispatch header specializes `DeviceOperationInstanceFactory` on the full
`DeviceGroupedConvFwdMultipleABD<..., specific_ops>` type and dispatches to the appropriate
`add_device_*` functions.

### 4.2 CK Source Locations

| Directory | Files | Description |
|-----------|-------|-------------|
| `grouped_conv3d_fwd_bilinear/` | ~10 | 3D forward bilinear |
| `grouped_conv3d_fwd_scale/` | ~10 | 3D forward scale |
| `grouped_conv3d_fwd_scaleadd_scaleadd_relu/` | ~5 | 3D forward fused |
| `grouped_conv2d_fwd_clamp/` | ~10 | 2D forward clamp |
| `grouped_conv3d_fwd_clamp/` | ~10 | 3D forward clamp |
| `grouped_conv2d_fwd_bias_clamp/` | ~10 | 2D forward bias+clamp |
| `grouped_conv3d_fwd_bias_clamp/` | ~10 | 3D forward bias+clamp |
| **Total** | **~65** | (estimate) |

> **Note:** Additional fused directories exist in CK (convinvscale, convscale, convscale_add,
> convscale_relu, dynamic_op, bias_bnorm_clamp, scaleadd_ab) but are NOT referenced by
> `DeviceOperationInstanceFactory_uses.md`. These can be deferred as future work.

### 4.3 Work Items

1. Create factory dispatch headers (5 files)
2. Create `.inc` files for each dispatch header
3. Convert `.cpp` source files from CK directories
4. Update CMakeLists.txt
5. Factory instance headers for any missing variants
   - Check if `device_grouped_conv_fwd_xdl_outelementop_instance.hpp` and
     `device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp` are already converted
   - If not, convert them following the `GROUPED_CONV_FWD_CONVERSION_PLAN.md` pattern

### 4.4 Checklist

| Item | Status |
|------|--------|
| Dispatch: `grouped_convolution_forward_bilinear.hpp` | [ ] |
| Dispatch: `grouped_convolution_forward_scale.hpp` | [ ] |
| Dispatch: `grouped_convolution_forward_scaleadd_scaleadd_relu.hpp` | [ ] |
| Dispatch: `grouped_convolution_forward_clamp.hpp` | [ ] |
| Dispatch: `grouped_convolution_forward_bias_clamp.hpp` | [ ] |
| `.inc` files for each dispatch header | [ ] |
| .cpp files for all fused directories | [ ] |
| CMakeLists.txt updated | [ ] |
| Missing factory instance headers (if any) | [ ] |

---

## Phase 5: Non-Grouped Convolutions

**Covers entries:** #6 (conv fwd), #7 (conv bwd data)

These are simpler non-grouped convolutions with fixed NHWC/KYXC/NHWK layout, 2D only.

### 5.1 CK Template Classes

| CK Template Class | Direction | CK Factory Header |
|---|---|---|
| `DeviceConvFwd` | Forward | `convolution_forward.hpp` |
| `DeviceConvBwdData` | Backward Data | `convolution_backward_data.hpp` |

### 5.2 CK Source Locations

| Directory | Files | Description |
|-----------|-------|-------------|
| `conv2d_fwd/` | ~5 | Non-grouped 2D forward |
| `conv2d_bwd_data/` | ~5 | Non-grouped 2D backward data |
| **Total** | **~10** | |

### 5.3 Approach: Separate Instance Data

Create separate instance data helpers specifically for non-grouped conv. This maintains structural
parity with the original CK library, keeping the non-grouped path distinct from the grouped path.
The underlying kernels used may also differ between grouped and non-grouped convolutions, further
justifying separate instance data.

### 5.4 Work Items

1. Create instance data helpers for non-grouped conv (separate from grouped conv helpers)
2. Create factory dispatch headers:
   - `convolution_forward.hpp` - specializes `DeviceOperationInstanceFactory<DeviceConvFwd<...>>`
   - `convolution_backward_data.hpp` - specializes `DeviceOperationInstanceFactory<DeviceConvBwdData<...>>`
3. Convert `.cpp` source files
4. Update CMakeLists.txt

### 5.5 Checklist

| Item | Status |
|------|--------|
| Instance data helpers (non-grouped conv) | [ ] |
| Dispatch: `convolution_forward.hpp` | [ ] |
| Dispatch: `convolution_backward_data.hpp` | [ ] |
| .cpp files: `conv2d_fwd/` (~5 files) | [ ] |
| .cpp files: `conv2d_bwd_data/` (~5 files) | [ ] |
| CMakeLists.txt updated | [ ] |

---

## Solver File Integration

After each phase, the corresponding MIOpen solver files need to be updated to use the ck_builder
factories behind `#ifdef CK_EXPERIMENTAL_BUILDER`. This follows the same pattern as the existing
2D forward integration in `conv_hip_implicit_gemm_grouped_fwd_xdlops.cpp`.

### Files to Update

| Solver File | Phases | Factories Used |
|------------|--------|----------------|
| `src/include/miopen/solver/implicitgemm_ck_util.hpp` | 1, 2 | DeviceOpGBwdPtrs (bwd data), DeviceOpGWrwPtrs (wrw), DeviceOpGBwdWeightDefaultPtrs (3D wrw), DeviceOpGBwdWeightBilinearPtrs (3D wrw bilinear), DeviceOpGBwdWeightScalePtrs (3D wrw scale) |
| `src/solver/conv/conv_hip_implicit_gemm_fwd_xdlops.cpp` | 5 | DeviceOpPtrs (non-grouped fwd) |
| `src/solver/conv/conv_hip_implicit_gemm_bwd_data_xdlops.cpp` | 5 | DeviceOpBwdPtrs (non-grouped bwd) |
| `src/solver/conv/conv_hip_implicit_gemm_grouped_wrw_xdlops.cpp` | 2 | DeviceOpGWrwPtrs (2D wrw) |
| `src/solver/conv/conv_hip_implicit_gemm_3d_grouped_fwd_xdlops.cpp` | 2, 3, 4 | DeviceOpGFwdBilinearPtrs (3D fwd bilinear), DeviceOpGFwdScalePtrs (3D fwd scale), DeviceOpGFwdDefaultPtrs (3D fwd default), plus WrW aliases |
| `src/solver/conv/conv_hip_implicit_gemm_3d_grouped_bwd_xdlops.cpp` | 1 | DeviceOpGBwdBilinearPtrs (3D bwd bilinear), DeviceOpGBwdScalePtrs (3D bwd scale), DeviceOpGBwdDefaultPtrs (3D bwd default) |
| `src/solver/conv_ck_igemm_fwd_bias_res_add_activ_fused.cpp` | 4 | DeviceOp (3D fwd scaleadd_scaleadd_relu) |
| `src/solver/conv_ck_igemm_grp_fwd_activ_fused.cpp` | 4 | DeviceOpGFwdActPtrs (N-D fwd clamp) |
| `src/solver/conv_ck_igemm_grp_fwd_bias_activ_fused.cpp` | 4 | DeviceOpGFwdBiasActivPtrs (N-D fwd bias_clamp) |

---

## Directory Structure (After All Phases Complete)

```
src/kernels/ck_builder/
├── include/miopen/ck_builder/
│   ├── instance_data/
│   │   ├── common.hpp                         (existing)
│   │   ├── xdl.hpp                            (existing - fwd)
│   │   ├── xdl_v3.hpp                         (existing - fwd)
│   │   ├── xdl_large_tensor.hpp               (existing - fwd)
│   │   ├── wmma.hpp                           (existing - fwd)
│   │   ├── wmma_v3.hpp                        (existing - stub)
│   │   ├── wmma_v3_large_tensor.hpp           (existing - stub)
│   │   ├── dl.hpp                             (existing - fwd)
│   │   ├── xdl_bwd_data.hpp                   (NEW - Phase 1)
│   │   ├── wmma_bwd_data.hpp                  (NEW - Phase 1)
│   │   ├── wmma_v3_bwd_data.hpp               (NEW - Phase 1, stub)
│   │   ├── xdl_bwd_weight.hpp                 (NEW - Phase 2)
│   │   ├── xdl_two_stage_bwd_weight.hpp       (NEW - Phase 2)
│   │   ├── wmma_bwd_weight.hpp                (NEW - Phase 2)
│   │   └── dl_bwd_weight.hpp                  (NEW - Phase 2)
│   ├── factories/
│   │   ├── device_operation_instance_factory.hpp   (existing)
│   │   ├── grouped_convolution_forward.hpp         (existing)
│   │   ├── grouped_convolution_forward_bilinear.hpp      (NEW - Phase 4)
│   │   ├── grouped_convolution_forward_scale.hpp         (NEW - Phase 4)
│   │   ├── grouped_convolution_forward_scaleadd_scaleadd_relu.hpp (NEW - Phase 4)
│   │   ├── grouped_convolution_forward_clamp.hpp         (NEW - Phase 4)
│   │   ├── grouped_convolution_forward_bias_clamp.hpp    (NEW - Phase 4)
│   │   ├── grouped_convolution_backward_data.hpp         (NEW - Phase 1)
│   │   ├── grouped_convolution_backward_data_bilinear.hpp (NEW - Phase 1)
│   │   ├── grouped_convolution_backward_data_scale.hpp    (NEW - Phase 1)
│   │   ├── grouped_convolution_backward_weight.hpp        (NEW - Phase 2)
│   │   ├── grouped_convolution_backward_weight_bilinear.hpp (NEW - Phase 2)
│   │   ├── grouped_convolution_backward_weight_scale.hpp    (NEW - Phase 2)
│   │   ├── convolution_forward.hpp                  (NEW - Phase 5)
│   │   ├── convolution_backward_data.hpp            (NEW - Phase 5)
│   │   ├── *.inc                                    (existing + NEW)
│   │   ├── grouped_conv_fwd/                        (existing - 24 headers)
│   │   ├── grouped_conv_bwd_data/                   (NEW - Phase 1, ~9 headers)
│   │   └── grouped_conv_bwd_weight/                 (NEW - Phase 2, ~12 headers)
│   ├── kernel_instantiation.hpp                (existing)
│   └── shared.hpp                              (existing)
├── factories/
│   ├── grouped_conv2d_fwd/                     (existing - 96 .cpp)
│   ├── grouped_conv3d_fwd/                     (NEW - Phase 3, ~74 .cpp)
│   ├── grouped_conv1d_fwd/                     (NEW - Phase 3, ~4 .cpp)
│   ├── grouped_conv3d_fwd_bilinear/            (NEW - Phase 4)
│   ├── grouped_conv3d_fwd_scale/               (NEW - Phase 4)
│   ├── grouped_conv3d_fwd_scaleadd_scaleadd_relu/ (NEW - Phase 4)
│   ├── grouped_conv2d_fwd_clamp/               (NEW - Phase 4)
│   ├── grouped_conv3d_fwd_clamp/               (NEW - Phase 4)
│   ├── grouped_conv2d_fwd_bias_clamp/          (NEW - Phase 4)
│   ├── grouped_conv3d_fwd_bias_clamp/          (NEW - Phase 4)
│   ├── grouped_conv2d_bwd_data/                (NEW - Phase 1, ~39 .cpp)
│   ├── grouped_conv3d_bwd_data/                (NEW - Phase 1, ~40 .cpp)
│   ├── grouped_conv3d_bwd_data_bilinear/       (NEW - Phase 1, ~6 .cpp)
│   ├── grouped_conv3d_bwd_data_scale/          (NEW - Phase 1, ~6 .cpp)
│   ├── grouped_conv2d_bwd_weight/              (NEW - Phase 2, ~66 .cpp)
│   ├── grouped_conv3d_bwd_weight/              (NEW - Phase 2, ~61 .cpp)
│   ├── grouped_conv3d_bwd_weight_bilinear/     (NEW - Phase 2, ~7 .cpp)
│   ├── grouped_conv3d_bwd_weight_scale/        (NEW - Phase 2, ~7 .cpp)
│   ├── conv2d_fwd/                             (NEW - Phase 5, ~5 .cpp)
│   └── conv2d_bwd_data/                        (NEW - Phase 5, ~5 .cpp)
└── static_tests/                               (existing + NEW tests)
```

---

## Implementation Order and Dependencies

```
Phase 1 (Bwd Data)     Phase 2 (Bwd Weight)
    │                       │
    ├─ instance_data/       ├─ instance_data/
    ├─ factory headers      ├─ factory headers
    ├─ dispatch headers     ├─ dispatch headers
    ├─ .cpp files           ├─ .cpp files
    └─ solver integration   └─ solver integration

Phase 3 (3D/1D Fwd)    Phase 4 (Fwd Fused)     Phase 5 (Non-Grouped)
    │                       │                        │
    ├─ .cpp files only      ├─ dispatch headers      ├─ instance_data (maybe)
    └─ CMakeLists           ├─ .cpp files            ├─ dispatch headers
                            └─ solver integration    ├─ .cpp files
                                                     └─ solver integration
```

**Phases 1, 2, and 3 are independent** and can be worked on in parallel.
**Phase 4** depends on Phase 3 (3D forward .cpp files should exist before fused variants).
**Phase 5** is independent but lowest priority.

## Summary

| Phase | New Instance Data | Factory Headers | .cpp Files | Dispatch Headers | Solver Files |
|-------|-------------------|-----------------|------------|------------------|--------------|
| 1 - Bwd Data | 3 | ~9 | ~91 | 3 + .inc | 2 |
| 2 - Bwd Weight | 4 | ~12 | ~141 | 3 + .inc | 3 |
| 3 - 3D/1D Fwd | 0 | 0 | ~78 | 0 | 0 |
| 4 - Fwd Fused | 0 | 0-8 | ~65 | 5 + .inc | 3 |
| 5 - Non-Grouped | 2 | ~2 | ~10 | 2 | 2 |
| **Total** | **~9** | **~23-31** | **~385** | **~13** | **~10** |
