# SageAttention V3 Forward Pass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement SageAttention V3 (SA3) forward pass on AMD gfx950 using MXFP4, including offline preprocessing kernels (unified Q/K/V quantization + smoothing + delta_s) and the main FMHA pipeline.

**Architecture:** SA3 pre-quantizes all of Q, K, V to MXFP4 (pk_fp4_t data + e8m0_t scale) offline before the FMHA kernel. The FMHA kernel adds a delta_s correction term (Q_mean × K^T) to s_acc after GEMM0, and uses two-level P quantization (Level 1: scale P̃ by `p_scale_factor=6.0f` to fill FP4 dynamic range; Level 2: cast_tile_mx for per-group MXFP4 quantization). No K permutation is needed—`BlockGemmMxARegBSmemCRegV1` handles it via coordinate transforms.

**Tech Stack:** HIP C++ (ROCm clang++), CK Tile ops (`BlockGemmMxARegBSmemCRegV1`, `cast_tile_mx`, `sweep_tile_span`, `block_tile_reduce`), gfx950-only (`CK_USE_NATIVE_MX_SUPPORT`).

---

## Key Algorithm Reference

### Memory Layout Convention

CK FMHA supports two layouts controlled by `i_perm`. The default (`i_perm=true`) is:
- Q, K: `[B, H, seqlen, hdim]` row-major, `stride = hdim` (seqlen is the slow dimension per head)
- V (row-major mode): `[B, H, seqlen_k, hdim_v]`, or column-major (transposed) `[B, H, hdim_v, seqlen_k]`

All preprocessing kernels use the same `[B, H, seqlen, hdim]` convention for Q and K input.

### Input tensors (all pre-quantized before kernel call)

`pk_fp4_t` packs 2 FP4 values along the **memory-contiguous (hdim) direction** for Q/K, and along the **seqlen_k direction** for transposed V. Scale granularity is always 32 along the packed dimension.

```
Q_hat:   [B, H, seqlen_q,   hdim/2]    pk_fp4_t  (hdim 方向打包, 2 fp4 per uint8)
Q_scale: [B, H, seqlen_q,   hdim/32]   e8m0_t    (每32个hdim元素一个scale)
K_hat:   [B, H, seqlen_k,   hdim/2]    pk_fp4_t
K_scale: [B, H, seqlen_k,   hdim/32]   e8m0_t
V_hat:   [B, H, hdim_v,     seqlen_k/2] pk_fp4_t  (先transpose, 再沿seqlen_k方向打包)
V_scale: [B, H, hdim_v,     seqlen_k/32] e8m0_t   (每32个seqlen_k元素一个scale)
delta_s: [B, H, num_q_blocks, seqlen_k] float32
```

> **Design constraint:** The Q block size used during preprocessing (`kM0` rows per block) **must exactly match** the `kM0` tile size of the FMHA pipeline's GEMM0. This ensures each Q-block's q_mean corresponds exactly to one FMHA iteration's s_acc tile row group.

### Preprocessing (offline, before FMHA kernel)

```
// K preprocessing (global smoothing):
k_mean[d]   = mean_{b,h,n}(K[b,h,n,d])             // global channel mean over all tokens
K_smooth    = K - k_mean                             // broadcast subtract along d
K_hat, K_scale = quantize_mx(K_smooth)              // MXFP4, packs along hdim

// Q preprocessing (per-Q-block smoothing):
per Q block q_i (rows m ∈ [i*kM0, (i+1)*kM0)):
  q_mean[b,h,i,d] = mean_m(Q[b,h,m,d])             // per-block channel mean (kM0 rows)
  Q_smooth = Q[block] - q_mean                       // broadcast subtract
  Q_hat[q_i], Q_scale[q_i] = quantize_mx(Q_smooth)  // MXFP4, packs along hdim

// delta_s = Q_mean @ K^T  (NOT in SageAttnPreprocessKernel)
// Computed separately via CK Tile GEMM after Q/K preprocessing:
//   input A: q_mean  [B*H, num_q_blocks, hdim]  float32
//   input B: K       [B*H, hdim, seqlen_k]       float32  (original K, transposed)
//   output:  delta_s [B*H, num_q_blocks, seqlen_k] float32
// Uses existing ck_tile batched GEMM kernel, no new kernel needed.

// V preprocessing (transpose then quantize):
V_T[b,h,d,n] = V[b,h,n,d]                          // transpose seqlen_k <-> hdim_v
V_hat, V_scale = quantize_mx(V_T)                   // MXFP4, packs along seqlen_k dim
```

### FMHA hot loop (per K tile)

```
// STAGE 1: GEMM0 = Q_hat × K_hat^T  (BlockGemmMxARegBSmemCRegV1)
s_acc = gemm_0(q_hat, k_hat, q_scale, k_scale)

// Add delta_s correction: per-column broadcast, ADDED to s_acc
// delta_s[b, h, q_block_idx, n_start : n_start+kN0]  shape [kN0]
constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {           // M 方向
    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {       // N 方向
        const auto n_idx = get_x_indices_from_distributed_indices(
            s_acc.get_tile_distribution(), make_tuple(idx0, idx1)).at(number<1>{});
        s_acc(make_tuple(idx0, idx1)) += delta_s_tile(n_idx);   // per-column broadcast
    });
});

// Standard online softmax (unchanged from existing MX pipeline)
m_new  = max(m_old, rowmax(s_acc))
rescale = exp(m_old - m_new)        // per-row scalar
O_acc  *= rescale                   // rescale 历史累加
l_acc   = l_acc * rescale + rowsum(exp(s_acc - m_new))
p_compute = exp(s_acc - m_new)      // P̃ ∈ (0,1], float32

// Two-level P quantization
// Level 1: 预放大至 FP4 满量程
//   cast_tile_mx 内部: scale = exp2(ceil(log2(max_abs / fp4_max)))
//   其中 fp4_max = 6.0f (E2M1 最大值), rcp_dst_max = 1/6 (硬编码于 cast_tile_mx)
//   因此传入 p_norm = P̃ * 6 时, max_abs/6 = P̃_max, e8m0 scale 恰好适配 P̃ 的分布
//   p_scale_factor 默认值 = 6.0f = fp4_max (使 P̃ 缩放至 FP4 满量程)
p_norm = p_compute * p_scale_factor  // Level 1: 放大 (default p_scale_factor=6.0f)
(P_hat, p_scale) = cast_tile_mx(p_norm)  // Level 2: per-32-element MXFP4 quantization

// STAGE 3: GEMM1 = P_hat × V_hat  (BlockGemmMxARegBSmemCRegV1)
// gemm1_result ≈ p_norm × V = P̃ × p_scale_factor × V  (含量化误差)
o_acc_tmp = gemm_1(P_hat, p_scale, V_hat, V_scale)

// Level 1 correction: 除以 p_scale_factor, 还原为 P̃ × V
// 目的: O_acc 的语义是 Σ_tiles(P̃ × V × rescale_factor), 不能带多余的 p_scale_factor 系数
const float inv_p_scale_factor = 1.0f / p_scale_factor;
constexpr auto o_spans = decltype(O_acc)::get_distributed_spans();
sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
        O_acc(make_tuple(idx0, idx1)) +=
            o_acc_tmp(make_tuple(idx0, idx1)) * inv_p_scale_factor;
    });
});
// 最终: O_acc += P̃ × V, 与标准 FA 语义一致
```

#### Level-1 correction 伪码变量说明

| 变量 | 含义 |
|------|------|
| `p_compute` | `exp(s_acc - m_new)`，标准 FA 的未归一化注意力权重，per-element float，范围 (0,1] |
| `p_scale_factor` | Level-1 预放大系数（运行时标量），默认 6.0f = FP4 E2M1 最大表示值 |
| `p_norm` | `p_compute × p_scale_factor`，传入 `cast_tile_mx` 的浮点张量 |
| `P_hat` | `cast_tile_mx` 输出的 pk_fp4_t 量化结果 |
| `p_scale` | `cast_tile_mx` 输出的 e8m0_t per-group scale，使 `P_hat × p_scale ≈ p_norm` |
| `o_acc_tmp` | GEMM1 输出，≈ `p_norm × V × V_scale` ≈ `P̃ × p_scale_factor × V`（量化误差内）|
| `inv_p_scale_factor` | `1 / p_scale_factor`，用于 Level-1 correction |
| `O_acc` | 跨 tile 累加的注意力输出，语义为 `Σ(P̃ × V × 各tile rescale因子)` |

**各行目的：**
1. `p_norm = p_compute * p_scale_factor` — 放大 P̃ 使其 max 接近 6，让 cast_tile_mx 内部的 `max_abs/6` 接近 1，从而 e8m0 scale ≈ 1（减少 power-of-2 向上舍入的相对损失）
2. `cast_tile_mx(p_norm)` — per-32-element MXFP4 量化：计算组内绝对值最大值，向上取整为 2 的幂次作为 e8m0 scale，再用 `__builtin_amdgcn_cvt_scalef32_pk_fp4_f32` 转换
3. `gemm_1(P_hat, p_scale, V_hat, V_scale)` — MX MFMA，内置 scale 乘法，输出 ≈ `p_norm × V`
4. `O_acc += o_acc_tmp * inv_p_scale_factor` — 消除 Level-1 放大，使本 tile 贡献为 `P̃ × V`，与标准 FA 的 `O_acc += P̃ × V × rescale` 一致（rescale 已在 `O_acc *= rescale` 中处理）

---

## Task 1: Add `SAGEATTN_V3` to the quant scale enum

**Files:**
- Modify: `include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp`
- Modify: `example/ck_tile/01_fmha/quant.hpp`

**Step 1: Add enum value to `BlockAttentionQuantScaleEnum`**

In `block_attention_quant_scale_enum.hpp`, after the `MX = 4` entry:
```cpp
    MX            = 4, // Microscaling
    SAGEATTN_V3   = 5, // SageAttention V3: MXFP4 with delta_s smoothing
```

Add specialization after the `MX` one:
```cpp
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::SAGEATTN_V3>
{
    static constexpr const char* name = "sageattnv3";
};
```

**Step 2: Mirror in `quant.hpp`**

In `quant.hpp`, after `mx = 4`:
```cpp
    mx            = 4, // Microscaling (MX)
    sageattnv3    = 5, // SageAttention V3
```

In `serialize()`:
```cpp
    else if(type == quant_scale_enum::sageattnv3)
        os << "sav3";
```

In `decode()`:
```cpp
    else if(str == "sav3" || str == "5")
    {
        info.type = quant_scale_enum::sageattnv3;
    }
```

**Step 3: Verify no build breakage**
```bash
cd /root/rocm-libraries/projects/composablekernel/build
cmake --build . --target ck_tile -j$(nproc) 2>&1 | tail -5
```
Expected: no errors

**Step 4: Commit**
```bash
git add include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp \
        example/ck_tile/01_fmha/quant.hpp
git commit -m "ck_tile/fmha: add SAGEATTN_V3 quant scale enum"
```

---

## Task 2: Add `FmhaFwdSageAttnV3` type config and update `fmha_fwd_args`

**Files:**
- Modify: `example/ck_tile/01_fmha/fmha_fwd.hpp`

**Step 1: Add the tag struct**

After `struct FmhaFwdMxFp4 {};`, add:
```cpp
struct FmhaFwdSageAttnV3
{
};
```

**Step 2: Add the type config specialization**

After the `FmhaFwdTypeConfig<FmhaFwdMxFp4>` specialization:
```cpp
template <>
struct FmhaFwdTypeConfig<FmhaFwdSageAttnV3>
{
    using QDataType             = ck_tile::pk_fp4_t;
    using KDataType             = ck_tile::pk_fp4_t;
    using VDataType             = ck_tile::pk_fp4_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using PDataType             = ck_tile::pk_fp4_t;
    using OaccDataType          = float;
    using ODataType             = float;

    using QScaleDataType = ck_tile::e8m0_t;
    using KScaleDataType = ck_tile::e8m0_t;
    using VScaleDataType = ck_tile::e8m0_t;
    using PScaleDataType = ck_tile::e8m0_t;

    static constexpr ck_tile::index_t kQKScaleGranularity = 32;
    static constexpr ck_tile::index_t kVScaleGranularity  = 32;
};
```

**Step 3: Add SA3-specific fields to `fmha_fwd_args`**

In the `fmha_fwd_args` struct, add after `v_descale_ptr`:
```cpp
    const void* delta_s_ptr = nullptr;     // [B, H, num_q_blocks, seqlen_k] float32
    // p_scale_factor: Level-1 P pre-scaling coefficient.
    // Default = 6.0f = FP4 E2M1 max value, so that P̃ ∈ (0,1] is scaled to (0,6],
    // filling the FP4 dynamic range and minimizing e8m0 rounding loss in cast_tile_mx.
    float       p_scale_factor = 6.0f;

    ck_tile::index_t stride_delta_s;        // stride along seqlen_k (innermost)
    ck_tile::index_t nhead_stride_delta_s;
    ck_tile::index_t batch_stride_delta_s;
    ck_tile::index_t q_block_stride_delta_s;
```

**Step 4: Commit**
```bash
git add example/ck_tile/01_fmha/fmha_fwd.hpp
git commit -m "ck_tile/fmha: add FmhaFwdSageAttnV3 type config and delta_s args"
```

---

## Task 3: Extend `FmhaFwdKernel` kargs for SA3

**Files:**
- Modify: `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`

**Step 1: Add `FmhaFwdSageAttnV3Kargs` struct**

After `struct FmhaFwdGroupMXKargs`:
```cpp
    // SageAttention V3: extends MX kargs with delta_s correction and p_scale_factor
    struct FmhaFwdCommonSageAttnV3Kargs : FmhaFwdCommonMXKargs
    {
        const float* delta_s_ptr = nullptr;
        float        p_scale_factor;          // default 6.0f = FP4 E2M1 max

        ck_tile::index_t stride_delta_s;
        ck_tile::index_t nhead_stride_delta_s;
        ck_tile::index_t q_block_stride_delta_s;
    };

    struct FmhaFwdBatchSageAttnV3Kargs : FmhaFwdCommonSageAttnV3Kargs
    {
        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
        ck_tile::index_t batch_stride_delta_s;
    };
```

**Step 2: Add `SAGEATTN_V3` branch in `MakeKargs()`**

Mirror the pattern used for `BlockAttentionQuantScaleEnum::MX`:
```cpp
    else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3)
    {
        FmhaFwdBatchSageAttnV3Kargs sav3_kargs;
        // populate q/k/v descale ptrs + strides from MX path
        // then add delta_s fields from a.delta_s_ptr, a.p_scale_factor, a.stride_delta_s etc.
        return sav3_kargs;
    }
```

**Step 3: Commit**
```bash
git add include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp
git commit -m "ck_tile/fmha: add SageAttnV3 kargs to FmhaFwdKernel"
```

---

## Task 4: Create the SA3 FMHA pipeline

**Files:**
- Create: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_sageattn_default_policy.hpp`
- Create: `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_sageattn.hpp`

### Step 1: Create the default policy (thin alias)

```cpp
#pragma once
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"

namespace ck_tile {
// SA3 pipeline reuses the same tiling policy as the MX pipeline.
using BlockFmhaPipelineQRKSVSSageAttnDefaultPolicy = BlockFmhaPipelineQRKSVSDefaultPolicy;
} // namespace ck_tile
```

### Step 2: Create the main pipeline

Base on `block_fmha_pipeline_qr_ks_vs.hpp`. Changes from the MX pipeline:

1. **Static assert:** `QScaleEnum` must be `SAGEATTN_V3`
2. **operator() signature:** add `DeltaSDramBlockWindow` and `p_scale_factor` parameter
3. **After GEMM0:** add delta_s broadcast to `s_acc` via `sweep_tile_span`
4. **Level 1 P scaling:** `p_norm = p_compute * p_scale_factor` before `cast_tile_mx`
5. **After GEMM1:** `O_acc += gemm1_result * (1.0f / p_scale_factor)`

The `if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)` branches become unconditional (SA3 always uses MXFP4). Everything else (online softmax, m/l update, O rescale) is identical to the existing MX pipeline.

**Step 3: Add to `include/ck_tile/ops/fmha.hpp`** alongside existing pipeline includes.

**Step 4: Commit**
```bash
git add include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_sageattn*.hpp \
        include/ck_tile/ops/fmha.hpp
git commit -m "ck_tile/fmha: add BlockFmhaPipelineQRKSVSSageAttn pipeline"
```

---

## Tasks 5-7: Preprocessing — unified `SageAttnPreprocessKernel`

> **Design decision:** Q, K, V preprocessing are implemented as **one unified kernel** (`SageAttnPreprocessKernel`) with a mode parameter, rather than three separate kernels. Reasons: (1) identical structural pattern; (2) reduces launch overhead; (3) easier parallel dispatch from host. Development is incremental: add Q logic first, verify, then K, then V.

**Files to create:**
- `include/ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp`
- `include/ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp`

### Unified kernel design

```cpp
enum class SageAttnPreprocessMode
{
    Q = 0,  // Q smoothing (per-block mean) + MXFP4 quantization + store q_mean
    K = 1,  // K smoothing (provided global k_mean) + MXFP4 quantization
    V = 2,  // V transpose + MXFP4 quantization (no smoothing)
};
```

**Shared Kargs (all modes):**
```cpp
struct SageAttnPreprocessKargs
{
    SageAttnPreprocessMode mode;

    // Input data (float32, [B, H, seqlen, hdim] for Q/K; [B, H, seqlen_k, hdim_v] for V)
    const float* src_ptr;
    index_t      seqlen;    // seqlen_q or seqlen_k depending on mode
    index_t      hdim;
    index_t      batch;
    index_t      nhead;
    index_t      stride_src;       // = hdim (row stride of src)
    index_t      nhead_stride_src;
    index_t      batch_stride_src;

    // Output: quantized data [B, H, seqlen, hdim/2] pk_fp4_t
    void*        dst_hat_ptr;
    // Output: scale [B, H, seqlen, hdim/32] e8m0_t
    void*        dst_scale_ptr;

    // Mode-specific fields (use union or optional pointers):
    const float* k_mean_ptr  = nullptr;  // [hdim] global K mean (mode=K only)
    float*       q_mean_ptr  = nullptr;  // [B,H,num_q_blocks,hdim] output (mode=Q only)
    index_t      kM0         = 0;        // Q block size, = FMHA kM0 (mode=Q only)
};
```

**Grid dispatch:**
```
mode=Q: (num_q_blocks, nhead, batch)   — one block per Q tile of size kM0
mode=K: (num_k_blocks, nhead, batch)   — one block per kN0-row K tile
mode=V: (num_seqk_blocks, nhead, batch) — one block per V tile (hdim_v rows × kN0 cols input)
```

### V preprocessing detail (transpose + quantize)

Input: `[B, H, seqlen_k, hdim_v]` row-major
Output:
- `V_hat: [B, H, hdim_v, seqlen_k/2]` pk_fp4_t (seqlen_k is now contiguous after transpose)
- `V_scale: [B, H, hdim_v, seqlen_k/32]` e8m0_t

Pipeline:
1. Load tile `[kN0, hdim_v]` from input (seqlen_k × hdim_v block)
2. Transpose to `[hdim_v, kN0]` using `shuffle_tile`
3. `cast_tile_mx` along the seqlen_k (now innermost) dimension
4. Store `V_hat [hdim_v, kN0/2]`, `V_scale [hdim_v, kN0/32]`

### Step-by-step development

**Step 5a: Q mode only**
Implement `SageAttnPreprocessKernel` with mode=Q:
- Per-block column-wise mean reduction (`block_tile_reduce` along M)
- Subtract mean, `cast_tile_mx` along hdim
- Store `Q_hat, Q_scale, q_mean`
- Build and test against CPU reference

**Step 5b: K mode**
Add mode=K branch:
- Load provided `k_mean [hdim]`
- Subtract from K tile, `cast_tile_mx` along hdim
- Store `K_hat, K_scale`
- Test

**Step 5c: V mode**
Add mode=V branch:
- Load V tile, `shuffle_tile` to transpose
- `cast_tile_mx` along seqlen_k direction
- Store `V_hat, V_scale` with transposed layout
- Test

**Step 5d: Commit unified kernel**
```bash
git add include/ck_tile/ops/sageattn_preprocess/
git commit -m "ck_tile/sageattn: add unified SageAttnPreprocessKernel (Q/K/V modes)"
```

---

## Task 6 (renumbered): Umbrella header

**Files:**
- Create: `include/ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp`

```cpp
#pragma once
#include "ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp"
#include "ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp"
```

Add this include to `include/ck_tile/ops.hpp`.

**Commit:**
```bash
git add include/ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp
git commit -m "ck_tile/sageattn: add sageattn_preprocess umbrella header"
```

---

## Task 7 (renumbered): CPU reference implementation

**Files:**
- Create: `include/ck_tile/host/reference/reference_sageattn_preprocess.hpp`

Functions (all CPU float32, nested loops):

```cpp
// Q: per-block mean subtraction
void reference_sageattn_q_smooth(Q, q_mean_out, q_smooth_out, B, H, seqlen_q, hdim, kM0);

// K: global mean subtraction (k_mean averaged over B×H×seqlen_k)
void reference_sageattn_k_smooth(K, k_mean_out, k_smooth_out, B, H, seqlen_k, hdim);

// delta_s = Q_mean @ K^T  (CPU reference, used to validate CK Tile GEMM output)
// q_mean: [B, H, num_q_blocks, hdim], K: [B, H, seqlen_k, hdim] (original unsmoothed K)
void reference_sageattn_delta_s(q_mean, K, delta_s_out, B, H, num_q_blocks, seqlen_k, hdim);

// V: transpose (no smoothing)
void reference_sageattn_v_transpose(V, V_T_out, B, H, seqlen_k, hdim_v);

// Full SA3 FMHA reference (float32, no quantization, uses delta_s)
void reference_sageattn_fwd(Q_smooth, K_smooth, V, delta_s, O_out, scale_s, ...);
```

**Commit:**
```bash
git add include/ck_tile/host/reference/reference_sageattn_preprocess.hpp
git commit -m "ck_tile/sageattn: add CPU reference for SA3 preprocessing"
```

---

## Task 8 (renumbered): Preprocessing unit test

**Files:**
- Create: `test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp`
- Modify: `test/ck_tile/fmha/CMakeLists.txt`

### Test structure

```cpp
class SageAttnPreprocessTest : public ::testing::TestWithParam<
    std::tuple<int,int,int,int,int>> {}; // B, H, seqlen_q, seqlen_k, hdim

TEST_P(SageAttnPreprocessTest, QMode)  { /* GPU mode=Q vs CPU reference */ }
TEST_P(SageAttnPreprocessTest, KMode)  { /* GPU mode=K vs CPU reference */ }
TEST_P(SageAttnPreprocessTest, VMode)  { /* GPU mode=V, check transposed layout */ }
// delta_s 由 CK Tile batched GEMM 计算 (q_mean @ K^T)，此 test 验证 GEMM 输出与 CPU reference 一致。
// 不需要自定义 kernel：调用现有 ck_tile GEMM 接口，传入 q_mean 和原始 K（float32），
// 与 reference_sageattn_delta_s() 的结果对比（atol=1e-5, rtol=1e-4）。
TEST_P(SageAttnPreprocessTest, DeltaS) { /* CK Tile GEMM(q_mean, K^T) vs CPU reference */ }

INSTANTIATE_TEST_SUITE_P(SmallShapes, SageAttnPreprocessTest, ::testing::Values(
    std::make_tuple(1, 1,  64,  64, 64),
    std::make_tuple(2, 8, 128, 256, 128),
    std::make_tuple(1, 4, 256, 512, 64)
));
```

### CMakeLists.txt addition

```cmake
if("gfx950" IN_LIST GPU_TARGETS)
    add_gtest_executable(test_ck_tile_fmha_sageattn_preprocess
        test_fmha_sageattn_preprocess.cpp)
    target_compile_definitions(test_ck_tile_fmha_sageattn_preprocess PRIVATE
        CK_USE_NATIVE_MX_SUPPORT=1)
    set_tests_properties(test_ck_tile_fmha_sageattn_preprocess PROPERTIES
        LABELS "SMOKE_TEST;test_ck_tile_fmha")
endif()
```

**Commit:**
```bash
git add test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp \
        test/ck_tile/fmha/CMakeLists.txt
git commit -m "ck_tile/sageattn: add preprocessing unit tests"
```

---

## Task 9 (renumbered): Codegen scripts

**Files:**
- Modify: `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`
- Modify: `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`

### cpp_symbol_map.py

```python
FWD_DTYPE_MAP = { ..., "sageattnv3": "FmhaFwdSageAttnV3" }
QSCALE_MAP    = { ..., "sageattnv3": "ck_tile::BlockAttentionQuantScaleEnum::SAGEATTN_V3" }
QSCALE_CHECK_MAP = { ..., "sageattnv3": "t.quant_type == quant_scale_enum::sageattnv3" }
```

### fmha_fwd.py

```python
DTYPE_BITS = { ..., "sageattnv3": 4 }
# In KernelComponentFactoryGfx950:
_DT_SAGEATTN_V3 = ("sageattnv3",)
# add to get_dtypes() return value
```

**Commit:**
```bash
git add example/ck_tile/01_fmha/codegen/cpp_symbol_map.py \
        example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py
git commit -m "ck_tile/sageattn: add sageattnv3 to codegen symbol maps"
```

---

## Task 10 (renumbered): Wire up runner and example

**Files:**
- Modify: `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`
- Modify: `example/ck_tile/01_fmha/example_fmha_fwd.cpp`

### fmha_fwd_runner.hpp

Add `FmhaFwdSageAttnV3` branch: allocate delta_s buffer, run preprocessing kernels (mode=Q, K, V), fill `fmha_fwd_args` with `delta_s_ptr`, `p_scale_factor`, and all strides.

### example_fmha_fwd.cpp

- Add `"sageattnv3"` to dtype dispatch table
- Add `--p_scale_factor` CLI argument (default 6.0f)

**Commit:**
```bash
git add example/ck_tile/01_fmha/fmha_fwd_runner.hpp \
        example/ck_tile/01_fmha/example_fmha_fwd.cpp
git commit -m "ck_tile/sageattn: wire sageattnv3 into example runner"
```

---

## Task 11 (renumbered): CMakeLists updates

**Files:**
- Modify: `example/ck_tile/01_fmha/CMakeLists.txt`
- Modify: `test/ck_tile/fmha/CMakeLists.txt`

```cmake
# example/CMakeLists.txt
if("gfx950" IN_LIST GPU_TARGETS)
    list(APPEND FMHA_FWD_DTYPES "sageattnv3")
endif()

# test/CMakeLists.txt - add sageattnv3 to add_gtest_fwd V_TYPES
set(V_TYPES "fp16" "bf16" "fp8bf16" "fp32" "mxfp8" "mxfp4" "sageattnv3")
set(CPP_TYPE_sageattnv3 "FmhaFwdSageAttnV3")
```

**Commit:**
```bash
git add example/ck_tile/01_fmha/CMakeLists.txt test/ck_tile/fmha/CMakeLists.txt
git commit -m "ck_tile/sageattn: register sageattnv3 in CMake"
```

---

## Task 12 (renumbered): End-to-end smoke test

**Prerequisite:** gfx950 GPU available.

```bash
# Build
cmake --build . --target test_ck_tile_fmha_sageattn_preprocess \
                          test_ck_tile_fmha_fwd_sageattnv3 -j$(nproc)
# Run preprocessing tests
./bin/test_ck_tile_fmha_sageattn_preprocess
# Run FMHA forward tests
./bin/test_ck_tile_fmha_fwd_sageattnv3
# Run example binary
./bin/example_fmha_fwd -b 2 -h 8 -s 512 -sk 512 -d 64 --type sageattnv3
```

---

## Tolerance Notes

| Stage | atol | rtol |
|-------|------|------|
| Q/K smoothing output (float32) | 1e-5 | 1e-4 |
| After FP4 quantization (dequantized) | 0.1 | 0.05 |
| Full FMHA output vs float32 reference | 0.2 | 0.1 |

---

## Architecture Guards

All SA3 code guarded by:
```cpp
#if defined(CK_USE_NATIVE_MX_SUPPORT)
// SA3 code
#endif
```
or via CMake `target_compile_definitions(... PRIVATE CK_USE_NATIVE_MX_SUPPORT)`.
