# XDL Grouped Conv Forward Instance Compile-Time Constraints

Collected rules encountered during WAN forward convolution kernel tuning (gfx950/MI350).
Each rule documents a `static_assert` or template constraint hit when constructing a new
`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` instance.

---

## Template Parameter Reference

For quick orientation, the full parameter list in positional order:

```
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial,                  // 1, 2, or 3
    ALayout, BLayout,             // e.g. NHWGC, GKYXC
    DsLayout, ELayout,            // Ds=Empty_Tuple for plain conv
    ADataType, BDataType,
    AccDataType,                  // typically F32
    CShuffleDataType,
    DsDataTypes, EDataType,
    AElementwiseOp, BElementwiseOp, CDEElementwiseOp,  // typically all PassThrough
    ConvForwardSpecialization,    // see §Specializations
    GemmSpecialization,           // typically GemmMNKPadding
    NumGemmKPrefetchStage,        // 1 = single buffer, 2 = double buffer
    BlockSize,                    // e.g. 64, 128, 256
    MPerBlock, NPerBlock, KPerBlock,
    AK1, BK1,                     // vector load width in K dimension
    MPerXDL, NPerXDL,             // XDL instruction tile (32x32 on gfx9)
    MXdlPerWave, NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_AK1,
    ABlockLdsAddExtraM,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_BK1,
    BBlockLdsAddExtraN,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CBlockTransferClusterLengths_MBlock_MWave_MPerXdl_NBlock_NWave_NPerXdl,
    CBlockTransferScalarPerVector_NWaveNPerXdl
>
```

---

## Specializations

```cpp
ConvFwdDefault     // General case, any filter size
ConvFwd1x1P0       // 1x1 filter, with padding
ConvFwd1x1S1P0     // 1x1 filter, stride=1, no padding
ConvFwd3x3S1D1P0   // 3x3 filter, stride=1, dilation=1, no padding (NHWGC only)
ConvFwdOddC        // Odd input channel count (C not divisible by vector width)
```

---

## Collected Constraints

<!-- Add new rules below as they are encountered. Format:
### Rule N: <short title>
**Parameters affected:** ...
**Rule:** ...
**Diagnosis:** set CK_LOGGING=1 and look for the failing check in the log.
**Example:** ...
-->

*No rules collected yet — add the first one when a compile error is encountered.*

---

## How to Add a Rule

When a newly constructed instance fails to compile or fails the applicability check at runtime:

1. Set `CK_LOGGING=1` before running to get log output explaining the failing check.
2. Find the `static_assert` or `if constexpr` check in
   `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`
   (or the underlying GEMM pipeline headers).
3. Record the rule here with the format above.
