# Phase 4: Forward Fused Variant Factories - Implementation Plan

## Context

Phase 4 of the CK Builder conversion converts 5 fused forward convolution operation factories from CK's compile-time template specialization pattern to the CK Builder's constexpr function-based instance generation framework. Currently, solvers like `conv_hip_implicit_gemm_3d_grouped_fwd_xdlops.cpp` include the CK library's factory dispatch headers directly. After conversion, they'll use `MetaDeviceOperationInstanceFactory` to switch between CK and CK Builder factories.

## Scope

| Operation | NumDTensor | Spatial Dims | Layouts | Data Types |
|-----------|-----------|-------------|---------|------------|
| Bilinear | 1 | 3D only | NDHWGC/GKZYXC/NDHWGK | BF16, F16, F32, TF32, INT8 |
| Scale | 0 | 3D only | NDHWGC/GKZYXC/NDHWGK | BF16, F16, F32, TF32, INT8 |
| ScaleAddScaleAddRelu | 2 | 3D only | NDHWGC/GKZYXC/NDHWGK | F16 only |
| Clamp | 0 | 2D+3D | NHWGC/GKYXC/NHWGK + NDHWGC/GKZYXC/NDHWGK | BF16, F16, F32, TF32 |
| AddClamp (BiasClamp) | 1 | 2D+3D | NHWGC/GKYXC/NHWGK + NDHWGC/GKZYXC/NDHWGK | BF16, F16, F32, TF32 |

## Key Architectural Decisions

### Reuse Strategy
- **Scale** (NumDTensor=0): Reuse existing base XDL factory functions from `device_grouped_conv_fwd_xdl_instance.hpp` by passing `output_op=ckb::ElementwiseOperation::SCALE`. No new factory instance header needed.
- **Clamp** (NumDTensor=0): Same reuse approach with `output_op=ckb::ElementwiseOperation::CLAMP`.
- **Bilinear** (NumDTensor=1): Already has dedicated factory instance header `device_grouped_conv_fwd_xdl_bilinear_instance.hpp` with hardcoded `Bilinear` output_op.
- **AddClamp** (NumDTensor=1): Needs new factory instance headers OR refactor bilinear to accept output_op parameter. Recommend new dedicated headers.
- **ScaleAddScaleAddRelu** (NumDTensor=2): Needs new factory instance headers with NumDTensor=2.

### Reference Files
- **CK original bilinear dispatch**: `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp`
- **CK original scale dispatch**: `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp`
- **CK original clamp dispatch**: `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp`
- **Existing CK Builder PassThrough dispatch**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_convolution_forward.hpp`
- **Existing bilinear factory instances**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_bilinear_instance.hpp`
- **Existing base XDL factory instances**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp`
- **Common aliases**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp`
- **Existing 3D test**: `test/gtest/ck_builder_grouped_fwd_conv3d.cpp`
- **CMakeLists**: `src/kernels/ck_builder/CMakeLists.txt`

---

## Step 0: Update common_aliases.hpp

Add missing elementwise operation aliases needed by fused operations:

**File**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp`

Add after the existing `Bilinear` alias:
```cpp
[[maybe_unused]] constexpr auto Scale                = ckb::ElementwiseOperation::SCALE;
[[maybe_unused]] constexpr auto Clamp                = ckb::ElementwiseOperation::CLAMP;
[[maybe_unused]] constexpr auto AddClamp             = ckb::ElementwiseOperation::ADD_CLAMP;
[[maybe_unused]] constexpr auto ScaleAddScaleAddRelu = ckb::ElementwiseOperation::SCALE_ADD_SCALE_ADD_RELU;
```

(Verify exact enum names in `ck_tile/builder/types.hpp`)

---

## Step 1: Bilinear (3D Fwd, NumDTensor=1)

### 1a. Create dispatch header
**File**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_convolution_forward_bilinear.hpp`

Pattern: Mirror `grouped_convolution_forward_bilinear.hpp` from CK library (composablekernel).
- Specialize `DeviceOperationInstanceFactory<DeviceGroupedConvFwdMultipleABD<..., Bilinear, ...>>`
- Template params: NumDimSpatial, InLayout, WeiLayout, DLayouts, OutLayout, InDataType, WeiDataType, DDataTypes, OutDataType, ComputeType
- GetInstances() dispatches on: NumDimSpatial==3, NDHWGC/GKZYXC/NDHWGK, DLayouts::Size()==1
- Conditionally includes .inc files for XDL and WMMA

### 1b. Create .inc files

**File**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_convolution_forward_bilinear_xdl.inc`

Declare functions:
```
add_device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_{bf16,f16,f32,f32_tf32,int8}_instances()
```

**File**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_convolution_forward_bilinear_wmma_cshufflev3.inc`

Declare functions (4 parts each for BF16, F16):
```
add_device_grouped_conv3d_fwd_wmma_cshufflev3_bilinear_ndhwgc_gkzyxc_ndhwgk_{bf16,f16}_instances_part{1..4}()
```

### 1c. Create .cpp source files

Under `src/kernels/ck_builder/grouped_conv3d_fwd_bilinear/xdl/`:
- `device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp`
- `device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp`
- `device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp`
- `device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.cpp`
- `device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_int8_instance.cpp`

Each calls the existing bilinear factory functions with `NumDTensor=1`, spatial_dim=3, appropriate layouts and data types.

Example pattern (bf16):
```cpp
#include <miopen/ck_builder/factories/grouped_convolution_forward_bilinear.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_bilinear_instance.hpp>

namespace ckb = ck_tile::builder;
using namespace factories::grouped_conv_fwd;

constexpr auto NDHWGC = ckb::TensorLayout::NDHWGC;
constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
constexpr auto NDHWGK = ckb::TensorLayout::NDHWGK;
constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;

void add_device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<...>& instances)
{
    add_device_operation_instances<
        device_grouped_conv_fwd_xdl_bilinear_bf16_instances<1>(
            3, NDHWGC, GKZYXC, {NDHWGK}, NDHWGK, ConvFwdDefault)>(instances);
}
```

### 1d. Update CMakeLists.txt

Add `GROUPED_CONV3D_FWD_BILINEAR_XDL_SOURCES` list and include in `ck_builder` library.

### 1e. Create test

**File**: `test/gtest/ck_builder_grouped_fwd_conv3d_bilinear.cpp`

Compare CK factory vs CK Builder factory output for Bilinear DeviceOp type using `compare_instance_vectors()`.

---

## Step 2: Scale (3D Fwd, NumDTensor=0)

### 2a. Create dispatch header
**File**: `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_convolution_forward_scale.hpp`

Pattern: Mirror CK's `grouped_convolution_forward_scale.hpp`.
- Specialize for `DeviceGroupedConvFwdMultipleABD<..., Scale, ...>`
- GetInstances() dispatches on NumDimSpatial==3, NDHWGC/GKZYXC/NDHWGK, DLayouts::Size()==0
- Includes `grouped_convolution_forward_scale_xdl.inc` and WMMA .inc

### 2b. Create .inc files

**File**: `grouped_convolution_forward_scale_xdl.inc`

Declare:
```
add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_{bf16,f16,f32,f32_tf32,int8}_instances()
```

**File**: `grouped_convolution_forward_scale_wmma_cshufflev3.inc`

Declare (4 parts each for BF16, F16):
```
add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_{bf16,f16}_instances_part{1..4}()
```

### 2c. Create .cpp source files

Under `src/kernels/ck_builder/grouped_conv3d_fwd_scale/xdl/`:
- 5 files (bf16, f16, f32, f32_tf32, int8)

These reuse the **base XDL factory functions** with `output_op=Scale`:
```cpp
add_device_operation_instances<
    device_grouped_conv_fwd_xdl_bf16_instances<0>(
        3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwdDefault, {}, Scale)>(instances);
```

### 2d. Update CMakeLists.txt, create test

Same pattern as Step 1.

---

## Step 3: Clamp (2D+3D Fwd, NumDTensor=0) - Deferred

Clamp is significantly more complex due to 2D+3D support with many sub-variants (16x16, large_tensor, merged_groups, comp, comp_2x, comp_part2, mem_intra, mem_inter, direct_load). This step involves ~40+ .cpp files and multiple .inc files. Recommend implementing after Bilinear and Scale are validated.

Key approach: Reuse base XDL factory functions with `output_op=Clamp`, same as Scale.

---

## Step 4: ScaleAddScaleAddRelu (3D Fwd, NumDTensor=2) - Deferred

F16-only, 3D only. Needs new factory instance headers with `NumDTensor=2`. Smallest file count but requires new instance data structures.

---

## Step 5: AddClamp/BiasClamp (2D+3D Fwd, NumDTensor=1) - Deferred

Similar complexity to Clamp. Needs new factory instance headers (like bilinear but with AddClamp output_op).

---

## Implementation Order

1. **Step 0**: Update `common_aliases.hpp` (prerequisite for all)
2. **Step 1**: Bilinear - XDL only first (simplest, factory headers already exist)
3. **Step 2**: Scale - XDL only first (validates base function reuse pattern)
4. Build and test Steps 1-2 before proceeding
5. Steps 3-5 follow in subsequent phases

## Verification

1. Build the `ck_builder` library: `cmake --build build --target ck_builder`
2. Run comparison tests:
   - `./build/bin/ck_builder_grouped_fwd_conv3d_bilinear`
   - `./build/bin/ck_builder_grouped_fwd_conv3d_scale`
3. Tests should show 0 differences between CK factory and CK Builder factory instance lists
