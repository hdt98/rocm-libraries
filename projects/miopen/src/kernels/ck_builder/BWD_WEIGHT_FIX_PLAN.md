# Phase 3 Backward Weight: Status & Remaining Work

## Original Issues (ALL RESOLVED)

The original plan documented three categories of bugs, all of which have been fixed:

1. **Regular WMMA used wrong algorithm concept** — FIXED. Created `wmma_v3_bwd_weight.hpp`
   with `WmmaV3BwdWeightAlgorithm` satisfying `BwdWmmaV3Algorithm`. Routes to
   `ConvBwdWeightWmmaV3Factory` and creates `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3`.

2. **Two-stage WMMA used wrong helper** — FIXED. Created `wmma_two_stage_bwd_weight.hpp`
   with `WmmaTwoStageBwdWeightAlgorithm` satisfying `BwdTwoStageWmmaV3Algorithm`.

3. **Two-stage XDL .cpp files were empty stubs** — FIXED. Created
   `xdl_two_stage_bwd_weight.hpp` with `XdlTwoStageBwdWeightAlgorithm` satisfying
   `BwdTwoStageXdlAlgorithm`, and populated all ~44 .cpp stubs.

## Bilinear/Scale Support (NEWLY ADDED)

The bilinear/scale backward weight variants required their own infrastructure:

### Problem
The bilinear/scale `.cpp` factory files use `DeviceGroupedConvBwdWeightMultipleD` (the
MultipleD variant), but the existing instance data helpers only create the non-MultipleD
`DeviceGroupedConvBwdWeight_Xdl_CShuffle` or `DeviceGroupedConvBwdWeight_Wmma_CShuffle`.
This is a fundamental type mismatch — the `BwdMultiDXdlAlgorithm` and
`BwdMultiDWmmaV3Algorithm` concepts require `specialization = MULTIPLE_D`.

### Solution: New Instance Data Helpers
Created two new instance data helpers:

1. **`instance_data/xdl_multi_d_bwd_weight.hpp`** — XDL Multi-D backward weight
   - `XdlMultiDBwdWeightAlgorithm` with `specialization = MULTIPLE_D`
   - Uses 4D block transfers (same as regular XDL bwd weight)
   - Does NOT have `max_transpose_transfer_*` fields
   - Routes to `ConvBwdWeightMultiDXdlFactory` → `DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle`
   - Helper: `DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle<NumDTensor>()`

2. **`instance_data/wmma_multi_d_bwd_weight.hpp`** — WMMA Multi-D backward weight
   - `WmmaMultiDBwdWeightAlgorithm` with `specialization = MULTIPLE_D`
   - Uses 3D block transfers + `block_gemm_pipeline`
   - Routes to `ConvBwdWeightMultiDWmmaV3Factory` → `DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3`
   - Helper: `DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<NumDTensor>()`

### Signature Convention: `ConvMultiDBwdWeightSignature`
For backward weight MultipleD, the CK Builder framework reads:
- Elementwise ops from `Sig.input`, `Sig.weight`, `Sig.output` tensor descriptors
- DsLayout/DsDataType from `Sig.output.operation.auxiliary_operand_configs`

The `ConvMultiDBwdWeightSignature` places:
- Bilinear/Scale operation on `weight.operation.elementwise_operation` (maps to WeiElementwiseOp)
- D tensor configs (layout + data type) on `output.operation.auxiliary_operand_configs` (maps to DsLayout/DsDataType)

The D tensor config is automatically derived from `wei_layout` and `wei_data_type`.

### Updated Instance Headers
- `device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp` → uses `XdlMultiDBwdWeightInstance`
- `device_grouped_conv_bwd_weight_xdl_scale_instance.hpp` → uses `XdlMultiDBwdWeightInstance`
- `device_grouped_conv_bwd_weight_wmma_bilinear_instance.hpp` → uses `WmmaMultiDBwdWeightInstance`
- `device_grouped_conv_bwd_weight_wmma_scale_instance.hpp` → uses `WmmaMultiDBwdWeightInstance`

## Build Status

### XDL Bilinear/Scale: COMPILES
All 10 XDL .cpp files (5 bilinear + 5 scale) compile successfully.

### WMMA Bilinear/Scale: CK LIBRARY BUG
The 4 WMMA Multi-D .cpp files fail due to a CK library bug in
`flush_cache.hpp:494` (`no member named 'p_a_grid'` in the WMMA V3 gridwise GEMM argument).
This is a pre-existing issue in the CK library's WMMA V3 Multi-D implementation,
not a problem with our ck_builder code. The fix must come from the CK library team.

## Remaining Gaps

| # | Item | Status | Priority |
|---|------|--------|----------|
| 1 | Bilinear dispatch header (`grouped_convolution_backward_weight_bilinear.hpp`) | NOT YET CREATED | Medium |
| 2 | Scale dispatch header (`grouped_convolution_backward_weight_scale.hpp`) | NOT YET CREATED | Medium |
| 3 | Bilinear/Scale `.inc` files (4 files) | NOT YET CREATED | Medium |
| 4 | Conv3D bilinear/scale test cases | NOT IN test file | Low |
| 5 | Static tests for bilinear/scale | NOT CREATED | Low |
| 6 | Solver file integration | NOT DONE | Low |
| 7 | WMMA Multi-D CK library `flush_cache.hpp` bug | BLOCKED on CK team | N/A |

Items 1-3 are needed for solver integration but don't block the factory `.cpp` compilation.
Items 4-6 are test/integration work that can be done after the core infrastructure is stable.
Item 7 requires a fix in the composablekernel library.

## Dispatcher Routing Reference

| Concept | Factory | CK Device Class |
|---------|---------|-----------------|
| `BwdXdlAlgorithm` (4D, generic) | `ConvBwdWeightXdlFactory` | `DeviceGroupedConvBwdWeight_Xdl_CShuffle` |
| `BwdWmmaV3Algorithm` (3D, generic) | `ConvBwdWeightWmmaV3Factory` | `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3` |
| `BwdTwoStageXdlAlgorithm` (3D, TWO_STAGE) | `ConvBwdWeightTwoStageXdlFactory` | `DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle` |
| `BwdTwoStageWmmaV3Algorithm` (TWO_STAGE) | `ConvBwdWeightTwoStageWmmaV3Factory` | `DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` |
| **`BwdMultiDXdlAlgorithm` (4D, MULTIPLE_D)** | **`ConvBwdWeightMultiDXdlFactory`** | **`DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle`** |
| **`BwdMultiDWmmaV3Algorithm` (3D, MULTIPLE_D)** | **`ConvBwdWeightMultiDWmmaV3Factory`** | **`DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3`** |

## Key Source Files

| What | Path |
|------|------|
| XDL Multi-D helper (NEW) | `instance_data/xdl_multi_d_bwd_weight.hpp` |
| WMMA Multi-D helper (NEW) | `instance_data/wmma_multi_d_bwd_weight.hpp` |
| XDL regular helper | `instance_data/xdl_bwd_weight.hpp` |
| WMMA V3 regular helper | `instance_data/wmma_v3_bwd_weight.hpp` |
| XDL two-stage helper | `instance_data/xdl_two_stage_bwd_weight.hpp` |
| WMMA two-stage helper | `instance_data/wmma_two_stage_bwd_weight.hpp` |
| Common types | `instance_data/common.hpp` |
