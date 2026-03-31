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

### Rule 1: WaveSize must equal warp_size (64 on gfx9)
**Parameters affected:** BlockSize, MPerBlock, NPerBlock, MPerXDL, NPerXDL, MXdlPerWave, NXdlPerWave
**Rule:** `MWaves = MPerBlock / (MXdlPerWave × MPerXDL)`, `NWaves = NPerBlock / (NXdlPerWave × NPerXDL)`, `WaveSize = BlockSize / (MWaves × NWaves)`. WaveSize must equal 64 (warp size on gfx9/gfx950).
**Implication:** `BlockSize = MWaves × NWaves × 64`.
**Example (invalid):** BlockSize=64, MPerBlock=128, NPerBlock=32, MPerXDL=16, NPerXDL=16, MXdlPerWave=4, NXdlPerWave=1 → MWave=2, NWave=2 → WaveSize=64/4=16 ≠ 64. Fails `IsValidGemmCompilationParameter` at runtime.

### Rule 2: ThreadClusterLengths product must equal BlockSize and cover the tile slice
**Parameters affected:** ABlockTransferThreadClusterLengths_AK0_M_AK1, BBlockTransferThreadClusterLengths_BK0_N_BK1
**Rule:** For `S<AK0_t, M_t, AK1_t>`: product `AK0_t × M_t × AK1_t = BlockSize`. The cluster must cover BlockSliceLengths `S<AK0, MPerBlock, AK1>` via `ClusterLengths × ThreadSliceLengths = BlockSliceLengths`. With `AK1_t=1` (standard pattern), `AK0_t ≤ AK0=KPerBlock/AK1`, `M_t = BlockSize/AK0_t`.
**Implication:** For B-side with `NPerBlock=32, BK0=2, BK1=8`: max threads = 2×32×1=64. So BlockSize ≤ 64 when NPerBlock=32, BK0=2.
**Example (compile error):** BlockSize=256, NPerBlock=32, KPerBlock=16, BK1=8 → BK0=2, max_B_threads=64 < 256.

### Rule 3: CDEBlockTransferScalarPerVector must divide NWave×NPerXDL
**Parameters affected:** CDEBlockTransferScalarPerVector_NWaveNPerXdl, NPerXDL, NXdlPerWave, NPerBlock
**Rule:** `NWave×NPerXDL % CDEScalarPerVector == 0`.
**Example (compile error):** NPerBlock=64, NXdlPerWave=2, NPerXDL=32 → NWave=1, NWave×NPerXDL=32. ScalarPerVector=8: 32%8=0 ✓. But if computed cluster `S<1,4,1,4>` (smaller tile), `At(3)=4`, `4%8≠0` → static_assert fails.

### Rule 4: Small K_gemm shapes are fundamentally bandwidth-limited
**Shape:** C=K=16, filter 1×1×1 (K_gemm=16, N_gemm=16, M_gemm=301392)
**Finding:** The existing instance `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>` achieves ~22 TFlops.
- `Filter1x1Stride1Pad0` is ~2.5× faster than `Filter1x1Pad0` for this shape.
- Larger BlockSize (128 vs 64) makes it worse (~20.7 TFlops) due to fewer blocks and less parallelism.
- The shape is limited by memory bandwidth; no XDL kernel variant meaningfully exceeds 22 TFlops.

---

## How to Add a Rule

When a newly constructed instance fails to compile or fails the applicability check at runtime:

1. Set `CK_LOGGING=1` before running to get log output explaining the failing check.
2. Find the `static_assert` or `if constexpr` check in
   `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`
   (or the underlying GEMM pipeline headers).
3. Record the rule here with the format above.
