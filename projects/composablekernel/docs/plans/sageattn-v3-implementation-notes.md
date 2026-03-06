# SageAttention V3 Forward Pass 实现说明

本文档详细解释 `40aa101c39`、`aede18aaf9`、`b3c2f40f58` 三个 commit 中所有改动的技术含义及其工作原理。

---

## 目录

1. [算法背景：SA3 与标准 FMHA 的区别](#1-算法背景)
2. [整体架构：四层抽象中 SA3 的位置](#2-整体架构)
3. [预处理 Kernel：`sageattn_preprocess`](#3-预处理-kernel)
4. [FMHA Pipeline：`BlockFmhaPipelineQRKSVSSageAttn`](#4-fmha-pipeline)
5. [FMHA Kernel：`FmhaFwdKernel` 的 SA3 分支](#5-fmha-kernel)
6. [数据类型系统：`FmhaFwdSageAttnV3`](#6-数据类型系统)
7. [Kargs 扩展：SA3 专属字段](#7-kargs-扩展)
8. [代码生成：12 个 kernel 实例](#8-代码生成)
9. [Example 层：runner 与 API 调用](#9-example-层)
10. [CPU Reference 实现](#10-cpu-reference-实现)
11. [`test_fmha_fwd` 中 SA3 的 CPU reference 验证路径](#11-test_fmha_fwd-中-sa3-的-cpu-reference-验证路径)
12. [单元测试：`test_fmha_sageattn_preprocess`](#12-单元测试test_fmha_sageattn_preprocess)
13. [E2E Bug 修复（第二个 commit）](#13-e2e-bug-修复)
14. [死代码清理（第三个 commit）](#14-死代码清理)
15. [关键工程细节汇总](#15-关键工程细节汇总)

---

## 1. 算法背景

### 标准 FMHA 的计算流程

标准 Flash Attention 的每个 Q tile 对应一个外循环迭代，流程如下：

```
for each K/V tile:
    S = Q @ K^T                  # GEMM0，fp16 × fp16 → float
    S' = S * scale_s             # 缩放
    P = softmax_online(S')       # 在线 softmax（含 m/l 更新）
    O += P @ V                   # GEMM1
O = O / l                        # 归一化
```

### SA3 的改动

SageAttention V3 的核心思路是把 Q/K/V 用 MXFP4（4 位浮点，microscaling）表示，但直接量化会有较大的误差。SA3 用两个手段来补偿精度损失：

**手段 1：均值平滑（Mean Smoothing）**

量化前先减去各 channel 的均值，使 Q/K 在数值上更接近零均值分布，从而减小 MXFP4 量化的相对误差。
- **Q**：用每个 Q block（kM0 行）的 channel mean（即列均值），记为 `Q_mean_block`
- **K**：用所有 token 的 global channel mean，记为 `K_mean_global`

**手段 2：delta_s 修正**

均值平滑后，注意力分数变成了 `Q_smooth @ K_smooth^T`，而不是原始的 `Q @ K^T`。两者之差恰好等于：

```
delta_s = Q_mean_block @ K^T
```

（这里利用了 `K_smooth = K - K_mean_global`，而 `K_mean_global` 项展开后会消去。详见 SageAttention V3 论文。）

因此，在 GEMM0 结束后，把 `delta_s` 加回到 `s_acc`，就能恢复精确的注意力分数。

**手段 3：两级 P 量化**

softmax 之后得到的 P 矩阵（元素值在 0~1 之间），直接转 MXFP4 会浪费精度（MXFP4 的 E2M1 格式最大值是 6.0，而 P 的值全部小于 1.0）。SA3 引入 `p_scale_factor = 6.0` 来填满动态范围：

```
P_norm = P * p_scale_factor        # Level-1：填满 FP4 range
P_fp4 = MXFP4_quantize(P_norm)    # Level-2：标准 MXFP4 量化
O_acc += P_fp4 @ V / p_scale_factor  # 最终除回，补偿 Level-1
```

### SA3 完整数据流

```
离线预处理（一次）：
  Q_fp32 → subtract Q_block_mean → MXFP4 quantize → Q_hat + Q_scale
  K_fp32 → subtract K_global_mean → MXFP4 quantize → K_hat + K_scale
  V_fp32 → transpose → MXFP4 quantize → V_hat + V_scale
  delta_s = Q_block_mean @ K_fp32^T → [B, H, num_q_blocks, seqlen_k] float32

在线 FMHA（每次 forward）：
  for each K/V tile:
    s_acc = MXFP4_GEMM(Q_hat, K_hat)          # 硬件 MX MFMA
    s_acc += delta_s_tile                       # 修正
    s_acc = s_acc * scale_s
    P = softmax_online(s_acc)
    P_norm = P * p_scale_factor
    P_fp4 = cast_tile_mx(P_norm)
    O_acc += MXFP4_GEMM(P_fp4, V_hat) * (1/p_scale_factor)
  O = O_acc / l
```

---

## 2. 整体架构

CK-Tile 的 FMHA 算子遵循四层层次结构。SA3 的每一层都有对应的实现：

```
┌─────────────────────────────────────────────────────────────────┐
│  Example / API 层                                                │
│  fmha_fwd_runner.hpp, fmha_fwd.hpp, example_fmha_fwd.cpp       │
│  → 负责参数解析、张量分配、调用 fmha_fwd() API                    │
├─────────────────────────────────────────────────────────────────┤
│  Kernel 层                                                       │
│  include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp            │
│  → 确定 grid/block，构造 delta_s tile window，分发到 pipeline     │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline 层（新增）                                              │
│  include/ck_tile/ops/fmha/pipeline/                             │
│    block_fmha_pipeline_qr_ks_vs_sageattn.hpp                   │
│  → 实现 SA3 主循环（GEMM0 + delta_s + softmax + 两级量化 + GEMM1）│
├─────────────────────────────────────────────────────────────────┤
│  Block/Warp 层（复用 MX 已有实现）                                │
│  BlockGemmMxARegBSmemCRegV1 + WarpGemmMfmaF16F16F32M*N*K*        │
│  → 底层 MXFP4 MFMA 指令封装                                      │
└─────────────────────────────────────────────────────────────────┘

另有独立的预处理层：
┌─────────────────────────────────────────────────────────────────┐
│  SageAttnPreprocessKernel                                        │
│  include/ck_tile/ops/sageattn_preprocess/                       │
│  → 离线量化 Q/K/V，输出 hat + scale + q_mean                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 预处理 Kernel

### 文件位置

```
include/ck_tile/ops/sageattn_preprocess/
  kernel/sageattn_preprocess_kernel.hpp    ← kernel 层（HIP 调度）
  pipeline/sageattn_preprocess_pipeline.hpp ← pipeline 层（实际计算）
  sageattn_preprocess.hpp                  ← 汇总 include
```

### 模式枚举

```cpp
enum class SageAttnPreprocessMode {
    Q = 0, // Q block 均值平滑 + MXFP4 量化 + 输出 q_mean
    K = 1, // 全局均值平滑 + MXFP4 量化
    V = 2, // 转置 + MXFP4 量化
};
```

三种模式用同一个 kernel template 实现，在 device 端通过 `kargs.mode` 分支。

### Grid 映射

```
mode=Q:  gridDim = (ceil(seqlen_q / kM0), nhead, batch)
         每个 CTA 处理一个 [kM0, hdim] 的 Q tile

mode=K:  gridDim = (ceil(seqlen_k / kN0), nhead, batch)
         每个 CTA 处理一个 [kN0, hdim] 的 K tile

mode=V:  gridDim = (hdim_v, nhead, batch)
         每个 CTA 处理 V 矩阵的一整行（即一个 hdim_v channel）
         输入 V 是 [seqlen_k, hdim_v]，输出 V_hat 是 [hdim_v, seqlen_k/2]
```

### Q 模式详解（`RunQ`）

Q 模式是三种中最复杂的，分四步：

**Step 1：清零共享内存列累加器**
```cpp
for(index_t d = tid; d < kCols; d += kBlockSize)
    smem_col_mean[d] = 0.0f;
block_sync_lds();
```
每个线程负责若干列，初始化为 0。

**Step 2：计算 column mean（列均值）**
```cpp
for(index_t d = tid; d < kCols; d += kBlockSize) {
    float sum = 0.0f;
    for(index_t r = 0; r < kRows; r++)
        sum += src_ptr[r * kCols + d];
    smem_col_mean[d] = sum / float(kRows);
}
block_sync_lds();
```
每个线程独立计算它负责列的行累加（无需 atomic，因为每列只有一个线程负责），然后同步到共享内存。

**Step 3：输出 q_mean**
```cpp
for(index_t d = tid; d < kCols; d += kBlockSize)
    q_mean_ptr[d] = smem_col_mean[d];
```
将共享内存中的列均值写到全局内存，供后续计算 delta_s 用。

**Step 4：逐行减均值 + MXFP4 量化**
```cpp
for(index_t r = tid; r < kRows; r += kBlockSize)
    QuantizeRowMXFP4(src_ptr + r*kCols, smem_col_mean,
                     dst_hat_ptr + r*(kCols/2),
                     dst_scale_ptr + r*(kCols/32), kCols);
```
行级并行：每个线程处理一行或多行（步长 kBlockSize）。

### K 模式详解（`RunK`）

K 模式比 Q 简单，不需要计算均值（global mean 在 CPU 上预先算好），直接逐行减均值量化：
```cpp
for(index_t r = tid; r < kRows; r += kBlockSize)
    QuantizeRowMXFP4(src + r*kCols, k_mean_ptr,
                     dst_hat + r*(kCols/2), dst_scale + r*(kCols/32), kCols);
```

### V 模式详解（`RunV`）

V 要做转置（[seqlen_k, hdim_v] → [hdim_v, seqlen_k]）并量化。每个 CTA 处理一个 `hdim_v` channel 的全部 seqlen_k 元素：

```cpp
// src_ptr 指向 V[0, hdim_channel]，步长是 hdim_v（列间距）
for(index_t g = tid; g < num_groups; g += kBlockSize) {
    // 加载 32 个不连续的元素（间隔 hdim_v）
    for(int j = 0; j < 32; j++)
        group_data[j] = src_ptr[(k_start + j) * hdim_v];
    // 计算 e8m0 scale
    float scale = compute_e8m0_scale(max_abs(group_data));
    dst_scale_ptr[g] = uint8_t(bit_cast<uint32_t>(scale) >> 23);
    // 打包 32 个 float → 16 字节 pk_fp4_t
    pack_fp4_group(hat_group, group_data, scale);
}
```

关键点：`src_ptr[(k_start+j) * hdim_v]` 是列访问（stride = hdim_v），实现了转置的效果，输出按行连续存储。

### MXFP4 量化核心（`QuantizeRowMXFP4`）

每组 32 个元素共享一个 e8m0 scale：

```cpp
// 1. 找最大绝对值
float max_abs = max(|x_i - mean_i|) for i in group

// 2. 计算 e8m0 scale（向上取到 2 的幂）
// e8m0 = float 的纯指数部分，上取整到最近的 2^k
float scale = exp2(ceil(log2(max_abs / 6.0f)))
// 实现：利用 float bit 操作
uint32_t bits = bit_cast<uint32_t>(max_abs * (1/6.0f));
bits = (bits + float_mant_mask) & float_head_mask;  // 清尾数，上取整指数
scale = bit_cast<float>(bits);

// 3. e8m0 编码：只存 float 的指数字节（bits[30:23]）
dst_scale[g] = uint8_t(bit_cast<uint32_t>(scale) >> 23);

// 4. 打包 FP4：每次处理 8 个 float，调用 4 次 AMDGCN 内建函数
// __builtin_amdgcn_cvt_scalef32_pk_fp4_f32 每次写入 2 个 FP4（pack 进 uint32_t 的 8 位）
for j in 0..3:
    x = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(x, data[2j], data[2j+1], scale, j);
// 得到的 uint32_t 包含 8 个 FP4 值（每个 4 bit）
```

---

## 4. FMHA Pipeline

### 文件位置

```
include/ck_tile/ops/fmha/pipeline/
  block_fmha_pipeline_qr_ks_vs_sageattn.hpp              ← 主 pipeline
  block_fmha_pipeline_qr_ks_vs_sageattn_default_policy.hpp ← policy（复用 MX）
```

### 静态断言

```cpp
static_assert(Problem::QScaleEnum == SAGEATTN_V3);
static_assert(std::is_same_v<VLayout, ColumnMajor>);  // V 必须是列主序
```

### operator() 签名

相比标准 MX pipeline，SA3 的 `operator()` 多出两个参数：

```cpp
const DeltaSDramBlockWindowTmp& delta_s_dram_block_window_tmp,  // [kM0, kN0] tile window
float p_scale_factor,                                            // 默认 6.0f
```

### 主循环逐步解析

#### 初始化阶段（循环前）

```cpp
// 一次性加载 Q tile 到寄存器（kQLoadOnce = true）
auto q_dram_window = make_tile_window(..., MakeQRegTileDistribution<Problem>());
auto q = load_tile(q_dram_window);
auto q_tile = tile_elementwise_in(q_element_func, q);  // identity

// 一次性加载 Q scale
auto q_scale = load_tile(q_scale_dram_window);

// 初始化 online softmax 状态
auto m = ...; set_tile(m, -inf);   // 行最大值
auto l = ...; clear_tile(l);       // 行和（softmax 分母）
auto o_acc = ...; clear_tile(o_acc);
```

#### 每次 KV 迭代 Step 1：GEMM0（Q_hat @ K_hat^T）

```cpp
// 分块 kK0 进行 MXFP4 GEMM0，Q/K 都用各自的 e8m0 scale
for(i_k0 in 0..k0_loops):
    auto q_slice = get_slice_tile(q_tile, ...);
    auto q_scale_slice = get_slice_tile(q_scale, ...);
    gemm_0(s_acc, q_slice, q_scale_slice, k_lds_window, k_scale_block_tile);
    // → BlockGemmMxARegBSmemCRegV1 调用 MFMA v_mfma_f32_32x32x16_fp4 等指令
```

`s_acc` 累加器是 float32，MFMA 指令把 4-bit 操作数解码为 float 后做乘加运算。

#### 每次 KV 迭代 Step 2：delta_s 修正

```cpp
// delta_s_tile 的形状是 [kM0, kN0]
// 但它在全局内存里只有 [1, seqlen_k]（M stride=0，后文详述）
// 所以 delta_s_tile[idx0, idx1] 对任何 idx0 都读同一行的 idx1 列
sweep_tile_span(s_spans[0], [&](auto idx0) {
    sweep_tile_span(s_spans[1], [&](auto idx1) {
        s_acc(idx0, idx1) += float(delta_s_tile[idx0, idx1]);
        // 实际上等价于 s_acc(idx0, idx1) += delta_s[idx1]
        // 广播效果：每行加同一个列偏移量
    });
});
```

这里用 `s_acc(i_j_idx) +=`（写 accessor）而非 `s_acc[i_j_idx]`（读 accessor），是因为 `operator[]` 返回 const 引用，不可赋值。

#### 每次 KV 迭代 Step 3：scale + online softmax

```cpp
tile_elementwise_inout([&scale_s](auto& x) { x *= scale_s; }, s_acc);

// Online softmax：找本 tile 行最大值
auto s = cast_tile<float>(s_acc);
auto m_local = block_tile_reduce(s, seq<1>{}, max, -inf);
block_tile_reduce_sync(m_local, max);

// 更新全局 m，计算 p_compute = exp(s - m)
sweep_tile_span(p_spans[0], [&](auto idx0) {
    auto tmp = exp(m_old[idx0] - m[idx0]);   // rescale 因子
    sweep_tile_span(p_spans[1], [&](auto idx1) {
        p_compute(idx0, idx1) = exp(s[idx0,idx1] - m[idx0]);
    });
    // 同步更新 o_acc 和 l
    o_acc(idx0, *) *= tmp;
    l(idx0) = tmp * l[idx0];
});
// 累加行和
l(idx0) += rowsum[idx0];
```

#### 每次 KV 迭代 Step 4：两级 P 量化

**Level-1：乘以 p_scale_factor**

```cpp
auto p_norm = ...; // 新 tile，float
sweep_tile_span(pn_spans[0], [&](auto idx0) {
    sweep_tile_span(pn_spans[1], [&](auto idx1) {
        p_norm(idx0, idx1) = p_compute(idx0, idx1) * p_scale_factor;
        // p_compute ∈ (0,1]，p_norm ∈ (0, 6.0]
        // FP4 E2M1 最大值 = 6.0，刚好填满动态范围
    });
});
```

**Level-2：cast_tile_mx（MXFP4 量化）**

```cpp
cast_tile_mx<kVScaleGranularity, WG::kAMLane>(p_result, p_scale_result, p_norm);
// kVScaleGranularity = 32：每 32 个 float 共享一个 e8m0 scale
// 内部对 p_norm 做与 QuantizeRowMXFP4 相同的 e8m0+FP4 量化
// p_result: pk_fp4_t tile
// p_scale_result: e8m0_t tile
```

#### 每次 KV 迭代 Step 5：GEMM1（P_fp4 @ V_hat）并补偿

```cpp
auto o_acc0 = ...; clear_tile(o_acc0);
gemm_1(o_acc0, p_slice, p_scale_slice, v_lds_window, v_scale_block_tile);

// Level-1 补偿：除以 p_scale_factor
const float inv_p = 1.0f / p_scale_factor;
sweep_tile_span(o0_spans[0], [&](auto idx0) {
    sweep_tile_span(o0_spans[1], [&](auto idx1) {
        o_acc(idx0, idx1) += o_acc0(idx0, idx1) * inv_p;
    });
});
```

为什么不能直接 `o_acc += o_acc0`？因为 `o_acc0` 是 `P_norm @ V = (P * 6) @ V`，所以实际贡献是 `6 × (P @ V)`，需要除以 6 才能得到正确的 `P @ V`。

#### 循环结束：归一化 O 和存 LSE

```cpp
// LSE = m + log(l)（数值稳定的 logsumexp）
lse(idx0) = m[idx0] * scale_s / C_LOG2E + log(l[idx0]);

// O 归一化
o_acc(idx0, idx1) *= (1.0f / l[idx0]);
```

---

## 5. FMHA Kernel

### `FmhaFwdKernel::run_()` 中 SA3 分支

SA3 的 kernel 分支在 `fmha_fwd_kernel.hpp` 约第 2099 行的 `if constexpr(QScaleEnum == SAGEATTN_V3)` 块中。主要工作是：

#### 构造 Q/K/V scale 的 tensor view

```cpp
// Q scale: [seqlen_q, hdim/32]，e8m0_t
const auto q_scale_dram = make_naive_tensor_view<global>(
    q_scale_ptr, make_tuple(seqlen_q, hdim/32),
    make_tuple(hdim/32, 1), ...);
// K scale: [seqlen_k, hdim/32]
// V scale: [hdim_v, seqlen_k/32]（列主序存储时的转置 scale）
```

#### 构造 delta_s 的广播 tensor view

这是整个实现中最关键的技巧：

```cpp
const float* delta_s_base =
    kargs.delta_s_ptr
    + i_batch * kargs.batch_stride_delta_s
    + i_nhead * kargs.nhead_stride_delta_s
    + i_tile_m * kargs.q_block_stride_delta_s;
// delta_s_base 现在指向当前 Q block 对应的那一行 [1, seqlen_k]

const auto delta_s_dram_naive = make_naive_tensor_view<global>(
    delta_s_base,
    make_tuple(number<kM0>{}, seqlen_k),     // 逻辑形状 [kM0, seqlen_k]
    make_tuple(number<0>{}, number<1>{}),     // M 方向 stride=0 ← 广播关键！
    ...);
```

**stride-0 广播原理**：`make_naive_tensor_view` 的 stride 参数控制访问 `[row, col]` 时的内存偏移量：
```
offset(row, col) = row * stride_M + col * stride_N
                 = row * 0       + col * 1
                 = col
```
无论 row 取哪个值，访问的都是 `delta_s_base[col]`，即物理上只有一行，但逻辑上像是 kM0 行完全相同的数据。

这样一来，pipeline 收到的 `delta_s_dram_window` 的 distribution 就和普通的 bias tile 完全一样，可以直接用 `MakeBiasDramTileDistribution` 加载，不需要特殊逻辑。

#### 调用 pipeline

```cpp
return FmhaPipeline{}(
    q_dram_window, identity{},
    k_dram_window, identity{},
    v_dram_window, identity{},
    bias_dram_window, identity{},      // 无 bias，窗口存在但不会被读
    randval_dram_window,               // 无 dropout
    lse_dram_window, identity{},
    identity{}, identity{}, identity{},
    mask, position_encoding,
    kargs.scale_s,
    variant, variant_params, block_indices,
    smem_ptr, dropout,
    q_scale_dram_window,               // SA3 新增
    k_scale_dram_window,               // SA3 新增
    v_scale_dram_window,               // SA3 新增
    delta_s_dram_window,               // SA3 新增
    kargs.p_scale_factor,              // SA3 新增
    sink_value);
```

---

## 6. 数据类型系统

### `FmhaFwdSageAttnV3` 类型别名

定义在 `fmha_fwd_runner.hpp` 中，聚合了 SA3 所有张量的数据类型：

```cpp
using FmhaFwdSageAttnV3 = FmhaFwdTypeConfig<
    pk_fp4_t,  // Q：MXFP4 packed format（每字节含 2 个 FP4）
    pk_fp4_t,  // K：同上
    pk_fp4_t,  // V：同上（列主序存储）
    float,     // P：softmax 中间结果（float32）
    float,     // O accumulator：GEMM1 输出
    float,     // LSE：logsumexp
    float,     // O output：最终输出
    e8m0_t,    // Q scale：e8m0 格式（8 bit，纯指数）
    e8m0_t,    // K scale
    e8m0_t,    // V scale
    pk_fp4_t,  // P scale（用于 cast_tile_mx 的两级量化）
    float      // delta_s（偏置修正项，float32）
>;
```

### `BlockAttentionQuantScaleEnum` 枚举

```cpp
enum class BlockAttentionQuantScaleEnum {
    NO_SCALE    = 0,
    PER_TENSOR  = 1,
    BLOCK_SCALE = 2,
    KV_BLOCKSCALE = 3,
    MX          = 4,
    SAGEATTN_V3 = 5,  // ← 新增
};
```

在编译期通过 `if constexpr(QScaleEnum == SAGEATTN_V3)` 选择不同的 kernel 路径，零运行时开销。

---

## 7. Kargs 扩展

### kargs 继承链

```
FmhaFwdCommonKargs          ← q/k/v/o ptr, seqlen, hdim, strides…
  └── FmhaFwdCommonMXKargs  ← q/k/v scale ptrs, scale strides…
        └── FmhaFwdCommonSageAttnV3Kargs  ← delta_s_ptr, p_scale_factor…
              └── FmhaFwdBatchSageAttnV3Kargs  ← batch_stride_*
```

### SA3 专属字段

```cpp
struct FmhaFwdCommonSageAttnV3Kargs : FmhaFwdCommonMXKargs {
    const float* delta_s_ptr;          // [B, H, num_q_blocks, seqlen_k]
    float        p_scale_factor;       // P 两级量化的 Level-1 缩放因子，默认 6.0f

    index_t stride_delta_s;            // 内层步长（沿 seqlen_k 方向），通常 = 1
    index_t nhead_stride_delta_s;      // head 间步长 = num_q_blocks * seqlen_k
    index_t q_block_stride_delta_s;    // Q block 间步长 = seqlen_k
};

struct FmhaFwdBatchSageAttnV3Kargs : FmhaFwdCommonSageAttnV3Kargs {
    index_t batch_stride_q_descale;    // 继承自 MX
    index_t batch_stride_k_descale;
    index_t batch_stride_v_descale;
    index_t batch_stride_delta_s;      // batch 间步长 = nhead * num_q_blocks * seqlen_k
};
```

### fmha_fwd.hpp 中的传参修复（第二个 commit）

`MakeKargsImpl` 是通用工厂函数，它不知道 SA3 的专属字段，所以这些字段在调用 `MakeKargsImpl` 之后必须手动设置：

```cpp
// MakeKargsImpl 处理 Q/K/V 的通用字段
auto kargs = MakeKargsImpl(...);

// SA3 专属字段必须在 MakeKargsImpl 之后单独设置
if constexpr(FmhaKernel::QScaleEnum == BlockAttentionQuantScaleEnum::SAGEATTN_V3) {
    kargs.delta_s_ptr            = reinterpret_cast<const float*>(args.delta_s_ptr);
    kargs.p_scale_factor         = args.p_scale_factor;
    kargs.stride_delta_s         = args.stride_delta_s;
    kargs.nhead_stride_delta_s   = args.nhead_stride_delta_s;
    kargs.batch_stride_delta_s   = args.batch_stride_delta_s;
    kargs.q_block_stride_delta_s = args.q_block_stride_delta_s;
}
```

如果不加这个块，`delta_s_ptr` 会是默认值 `nullptr`，kernel 访问空指针导致 GPU 段错误。

---

## 8. 代码生成

### 代码生成框架

SA3 的 kernel 实例由 Python 脚本动态生成，不是手写 C++：

```
example/ck_tile/01_fmha/codegen/
  ops/fmha_fwd.py          ← 核心：定义实例的维度参数和 pipeline 选择
  cpp_symbol_map.py        ← C++ 类型名映射
```

### SA3 的实例策略

```python
# fmha_fwd.py 中 KernelComponentFactoryGfx950 对 sageattnv3 的配置

# Tile 尺寸：只支持 hdim=128（kM0=128, kN0=128, kK0=64, kN1=128, kK1=64）
(128, 128): [FmhaFwdTileSize(128, 128, 64, 128, 64, 128, ...)]

# Pipeline 参数：固定 lse=True, dropout=False, bias=no
# 变化维度：mask（causal/no_mask）× padding（有/无）× group_mode（batch/group）
for mask, padding_config in ...:
    pipelines.append(FmhaFwdPipeline(
        "qr_sageattn",   # ← 选择 BlockFmhaPipelineQRKSVSSageAttn
        "col",           # V 列主序
        ...,
        qscale="sageattnv3",  # ← 生成 SAGEATTN_V3 枚举值
        ...
    ))
```

### 生成的 12 个实例

| 编号 | Padding | Mask | Group Mode |
|------|---------|------|------------|
| 1 | 无 padding | 无 mask | batch |
| 2 | 有 padding | 无 mask | batch |
| 3 | 无 padding | causal | batch |
| 4 | 有 padding | causal | batch |
| 5 | 无 padding | 无 mask | group |
| 6 | 有 padding | 无 mask | group |
| 7 | 无 padding | causal | group |
| 8 | 有 padding | causal | group |
| 9-12 | 同 1-4 + sink 变体 |  |  |

（实际数目取决于 mask_impl 参数展开，共生成 12 个 `.cpp` 文件）

### CMakeLists.txt 的 VERBATIM 修复

```cmake
# 修复前：--filter *sageattn* 中的 * 会被 ninja 的 shell 展开为文件列表
add_custom_command(...
    COMMAND python3 fmha_fwd.py --filter *sageattn*
    ...)

# 修复后：VERBATIM 告诉 CMake 直接把参数原样传给 shell，不做任何替换
add_custom_command(...
    COMMAND python3 fmha_fwd.py --filter \\*sageattn\\*
    ...
    VERBATIM)
```

`VERBATIM` 使得 `\\*sageattn\\*` 在生成 ninja 规则时变成字面量 `\*sageattn\*`，ninja 执行时把 `\*` 传给 shell 作为字面通配符，而不是让 shell 展开它。

---

## 9. Example 层

### Example Runner 的 SA3 特殊处理

**输入验证**（`fmha_fwd_runner.hpp`）：
```cpp
// SA3 是 MX 类型的子集，但不能被 MX 的 qscale 检查拦截
if(is_mx && !is_sageattnv3 && qscale.type != quant_scale_enum::mx)
    return fwd_result::invalid_args;  // 普通 MX 必须用 mx qscale

// SA3 暂时只实现 batch 模式（不支持 group 模式）
if(is_sageattnv3 && mode == mode_enum::group)
    return fwd_result::no_instance;  // SKIPPED，不是 FAILED

// SA3 只有 hdim=128 的 kernel
if(is_sageattnv3 && (hdim_q != 128 || hdim_v != 128))
    return fwd_result::no_instance;
```

**SA3 专属 args 填充**：
```cpp
if constexpr(is_sageattnv3) {
    // 生成 delta_s（= Q_mean @ K^T）
    // 分配 q_hat, q_scale, k_hat, k_scale, v_hat, v_scale, q_mean 缓冲区
    // 调用 sageattn_preprocess 量化 Q/K/V
    // 计算 delta_s = q_mean_fp32 @ k_fp32^T（用 CPU 或 rocBLAS）
    fmha_args.delta_s_ptr = delta_s_device_ptr;
    fmha_args.p_scale_factor = 6.0f;
    ...
}
```

### quant.hpp 中的字符串别名

```cpp
// 修复前：只识别 "sav3"
else if(str == "sav3" || str == "5")
    info.type = quant_scale_enum::sageattnv3;

// 修复后：额外接受 "sageattnv3"（用户友好名称）
else if(str == "sav3" || str == "sageattnv3" || str == "5")
    info.type = quant_scale_enum::sageattnv3;
```

---

## 10. CPU Reference 实现

### 文件位置

```
include/ck_tile/host/reference/reference_sageattn_preprocess.hpp
```

这个文件提供了 SA3 预处理和前向计算的**纯 float32 CPU 参考实现**，不涉及任何量化。其作用有三：

1. **验证正确性**：GPU kernel 的输出结果与 CPU reference 对比，确认数值吻合
2. **文档即代码**：用最直白的循环写法表达每个步骤的数学语义，便于阅读和审查
3. **生成 delta_s**：在 example runner 中，`delta_s` 这个修正项用 CPU reference 计算（不需要量化精度，float32 足够）

所有函数接受显式 stride 参数，统一使用 `[B, H, seqlen, hdim]` 行主序布局。

---

### `reference_sageattn_q_smooth`：Q 块均值平滑

```cpp
inline void reference_sageattn_q_smooth(
    const float* Q,          // 输入：[B, H, seqlen_q, hdim]
    float*       q_mean_out, // 输出：[B, H, num_q_blocks, hdim]，每个 Q block 的列均值
    float*       q_smooth_out, // 输出：[B, H, seqlen_q, hdim]，减去均值后的 Q
    int B, int H, int seqlen_q, int hdim,
    int q_block_size)        // = kM0，必须与 FMHA pipeline 的 Q tile 大小一致
```

**核心逻辑**（四层循环展开理解）：

```
对每个 (batch b, head h, Q-block qi):
  row_start = qi * q_block_size
  row_end   = min(row_start + q_block_size, seqlen_q)  ← 最后一块可能不满

  对每个 channel d:
    // 1. 计算这个 Q block 在 channel d 上的均值
    mean_d = mean(Q[b,h, row_start:row_end, d])

    // 2. 写出 q_mean：供后续计算 delta_s = q_mean @ K^T 使用
    q_mean_out[b, h, qi, d] = mean_d

    // 3. 对这个 Q block 的所有行减去 mean_d，写出平滑后的 Q
    for n in row_start..row_end:
        q_smooth_out[b, h, n, d] = Q[b, h, n, d] - mean_d
```

**关键设计决策**：均值是**按 Q block** 计算的，而不是全局均值。这样每个 Q block 有自己独立的均值向量，因此 delta_s 的形状是 `[B, H, num_q_blocks, seqlen_k]`——每个 Q block 对应一行 delta_s，而不是每个 query token 对应一行。这也是 GPU kernel 中 `q_block_stride_delta_s` stride 字段存在的原因。

**边界处理**：`row_end = min(row_start + kM0, seqlen_q)` 确保最后一个不满的 Q block 用实际行数计算均值，而不是 kM0，保证均值的数学正确性。

---

### `reference_sageattn_k_smooth`：K 全局均值平滑

```cpp
inline void reference_sageattn_k_smooth(
    const float* K,           // 输入：[B, H, seqlen_k, hdim]
    float*       k_mean_out,  // 输出：[hdim]，全局 channel 均值
    float*       k_smooth_out, // 输出：[B, H, seqlen_k, hdim]
    int B, int H, int seqlen_k, int hdim)
```

**核心逻辑**：

```
对每个 channel d:
  // 1. 计算跨所有 (batch, head, token) 的全局均值
  sum = Σ_{b,h,n} K[b, h, n, d]
  mean_d = sum / (B * H * seqlen_k)
  k_mean_out[d] = mean_d

  // 2. 对所有 K 元素减去 mean_d
  for b, h, n:
      k_smooth_out[b, h, n, d] = K[b, h, n, d] - mean_d
```

**与 Q 的关键区别**：K 的均值是**跨所有 batch、所有 head、所有 token** 的真正全局均值，形状只有 `[hdim]`。原因是 K 的量化（减均值 + MXFP4）在推理时是离线做的，全局均值只需要计算一次就对所有请求都有效；而 Q 是每次请求新来的，只能做 per-block 的局部均值。

`k_mean_out` 在 example runner 中会传入 GPU 的 `SageAttnPreprocessKernel` K 模式，作为 `k_mean_ptr` 参数。

---

### `reference_sageattn_delta_s`：计算修正项

```cpp
inline void reference_sageattn_delta_s(
    const float* q_mean,  // 输入：[B, H, num_q_blocks, hdim]，Q 平滑后的块均值
    const float* K,       // 输入：[B, H, seqlen_k, hdim]，原始未平滑的 K！
    float*       delta_s, // 输出：[B, H, num_q_blocks, seqlen_k]
    int B, int H, int num_q_blocks, int seqlen_k, int hdim)
```

**核心逻辑**（矩阵乘法展开）：

```
对每个 (b, h, qi, kj):
  delta_s[b, h, qi, kj] = dot(q_mean[b, h, qi, :], K[b, h, kj, :])
                         = Σ_d q_mean[b,h,qi,d] * K[b,h,kj,d]
```

这就是一个 batch 化的矩阵乘法：

```
delta_s[b, h] = q_mean[b, h]  @  K[b, h]^T
                [num_q_blocks, hdim]  ×  [hdim, seqlen_k]
              = [num_q_blocks, seqlen_k]
```

**为什么用原始 K 而不是 K_smooth**：数学推导如下。原始注意力分数为：

```
S = Q @ K^T
  = (Q_smooth + q_mean) @ (K_smooth + k_mean)^T
  = Q_smooth @ K_smooth^T   ← MXFP4 GEMM 计算这部分
  + q_mean @ K_smooth^T     ← 这部分
  + Q_smooth @ k_mean^T     ← 这部分（Q_smooth 零均值，期望为 0，近似忽略）
  + q_mean @ k_mean^T       ← 常数，对 softmax 没有影响（每行加同一个常数不改变结果）
```

实际实现中 `delta_s = q_mean @ K^T = q_mean @ (K_smooth + k_mean)^T`，展开后多出的 `q_mean @ k_mean^T` 项是每行（每个 kj）加同一个常数，对 softmax 结果没有影响，因此直接用原始 K 也是等价的，且更简单。

**形状说明**：`delta_s[b, h, qi, kj]` 中 `qi` 是 Q-block 的索引，不是单个 token 的索引。在 FMHA pipeline 中，同一个 Q block（kM0 行）里所有 token 共享同一行 delta_s，这就是为什么 kernel 里用 `stride-0` 广播技巧（M 方向步长为 0，kM0 行逻辑上指向同一片内存）。

---

### `reference_sageattn_v_transpose`：V 转置

```cpp
inline void reference_sageattn_v_transpose(
    const float* V,       // 输入：[B, H, seqlen_k, hdim_v]
    float*       V_T_out, // 输出：[B, H, hdim_v, seqlen_k]
    int B, int H, int seqlen_k, int hdim_v)
```

**核心逻辑**：

```
V_T[b, h, d, n] = V[b, h, n, d]
```

即把最后两个维度互换：原来是 `[seqlen_k, hdim_v]`（每行是一个 token 的 v 向量），转置后是 `[hdim_v, seqlen_k]`（每行是一个 hdim channel 的全部 token）。

**为什么转置**：FMHA 的 GEMM1 计算 `P @ V`，如果 V 是 `[seqlen_k, hdim_v]` 行主序，则 GEMM 读 V 时是按行读（K 维度 = seqlen_k），对 V 的访问是列访问（不连续），不利于向量化加载。转置后，V_T 是 `[hdim_v, seqlen_k]` 行主序，GEMM1 读 V_T 的 K 维度（seqlen_k）时变成了行内连续访问，利于 coalesced 全局内存读取。在 CK-Tile 中，V 的 `ColumnMajor` layout 标注正是对应这个转置后的内存排布。

---

## 11. `test_fmha_fwd` 中 SA3 的 CPU reference 验证路径

### 背景

`test_fmha_fwd_sageattnv3` 二进制对应的测试类型配置是 `FmhaFwdSageAttnV3`，其 `do_validation=1` 路径会在 CPU 上重算注意力输出，再与 GPU kernel 的输出做数值对比。整个 CPU 参考计算分布在 `fmha_fwd_runner.hpp` 中，按照 FMHA 的计算步骤依次展开，SA3 在多个步骤上有专属分支。

理解这条路径的前提是了解 SA3 的数据类型配置：

```cpp
// FmhaFwdTypeConfig<FmhaFwdSageAttnV3>
QDataType   = pk_fp4_t   // GPU 存储：MXFP4 packed（2个FP4压缩进1字节）
KDataType   = pk_fp4_t
VDataType   = pk_fp4_t
SaccDataType        = float   // GEMM0 累加器（QK 点积结果）
SMPLComputeDataType = float   // softmax 中间值
PDataType   = pk_fp4_t        // GPU 上 softmax 后的 P 矩阵
OaccDataType        = float   // GEMM1 累加器（PV 结果）
ODataType   = float           // 最终输出

QScaleDataType = KScaleDataType = VScaleDataType = e8m0_t  // MXFP4 的 e8m0 scale
kQKScaleGranularity = 32  // 每 32 个 Q/K 元素共享一个 scale
kVScaleGranularity  = 32  // 每 32 个 V 元素共享一个 scale
```

有一个关键的 **host 端 P 类型覆盖**：

```cpp
// fmha_fwd_runner.hpp:275
using PDataType = std::conditional_t<is_mx, float, typename TypeConfig::PDataType>;
```

SA3 属于 `is_mx=true`，所以 **host 端的 P 固定是 `float`，而不是 `pk_fp4_t`**。这是有意为之的设计（见下文 Step 4 的详细说明）。

---

### 验证流程全景

```
GPU输入数据（已量化）        CPU reference 计算步骤
──────────────────────       ──────────────────────────────────────────
Q_hat[nhead,sq,hdim/2]  ─→  Step 1: mx_descale(Q_hat, Q_scale) → Q_float[nhead,sq,hdim]
Q_scale[nhead,sq,hdim/32]
K_hat[nhead,sk,hdim/2]  ─→  Step 2: mx_descale(K_hat, K_scale) → K_float[nhead,sk,hdim]
K_scale[nhead,sk,hdim/32]
                         ─→  Step 3: GEMM0: S = Q_float @ K_float^T * scale_s
delta_s[B,nhead,nqblk,sk]─→  Step 4: S += delta_s[b,h,q_idx/kM0,k_idx] * scale_s  (SA3专属)
                         ─→  Step 5: masking（causal/window，所有类型共用）
                         ─→  Step 6: softmax(S) → P_float （host P 保持 float）
V_hat[nhead,hdim,sk/2]  ─→  Step 7: mx_descale(V_hat, V_scale) → V_float[nhead,hdim,sk]
V_scale[nhead,hdim,sk/32]
                         ─→  Step 8: GEMM1: O = P_float @ V_float
GPU输出 O_gpu[B,nhead,sq,hdim] ─→  Step 9: check_err(O_gpu, O_cpu, rtol=0.1, atol=0.26)
```

---

### Step 1 & 2：MXFP4 反量化（Q 和 K）

```cpp
// 提取当前 batch 对应的 scale tile
ck_tile::HostTensor<QScaleDataType> q_descale_host_ref({nhead, real_seqlen_q, hdim_q_scale});
// hdim_q_scale = hdim / kQKScaleGranularity = 128 / 32 = 4（每行 4 个 scale）

// 从全局 scale buffer 中抠出本 batch 的 tile（处理 batch/group 两种 mode 的 offset）
if(i_perm) q_descale_host_ref.ForEach([&](auto& self, auto i) {
    self(i) = q_descale_host(b_idx, i[0], i[1] + query_offset, i[2]);
});

// MXFP4 反量化：将 pk_fp4_t packed 数据还原为 float
auto q_host_ref2 = reference_batched_mx_descale<
    /*InType=*/  QDataType,       // pk_fp4_t
    /*ScaleType=*/QScaleDataType, // e8m0_t
    /*AccType=*/ SaccDataType,    // float
    /*OutType=*/ SaccDataType     // float
>(q_host_ref, q_descale_host_ref, kQKScaleGranularity /*=32*/);
```

`reference_batched_mx_descale` 的核心逻辑：
- 对每组 32 个 MXFP4 元素，取对应的 e8m0 scale 值 `s`
- e8m0 的实际数值 = `2^(s - 127)`（即把 8 位指数解码为 float）
- 每个元素的反量化结果 = `fp4_to_float(elem) * 2^(s-127)`

K 的处理完全对称，区别仅在 batch stride 的计算（需要考虑 GQA 的 `nr = nhead / nhead_k` 因子）。

此后 `q_host_ref2` 和 `k_host_ref2` 的形状均为 `[nhead, seqlen, hdim]`，元素类型为 `float`，代表的是量化后再反量化的 **近似值**（含量化误差），而非原始的 float Q/K。

---

### Step 3：QK GEMM（计算注意力分数矩阵 S）

```cpp
// s_host_ref 形状：[nhead, real_seqlen_q, real_seqlen_k]
reference_batched_gemm<
    /*AType=*/SaccDataType,       // float（反量化后的 Q）
    /*BType=*/SaccDataType,       // float（反量化后的 K）
    /*AccType=*/SaccDataType,     // float
    /*OutType=*/SMPLComputeDataType // float
>(q_host_ref2, k_host_ref2, s_host_ref,
  identity{}, identity{},
  scales(scale_s_host));          // 乘以 scale_s = 1/sqrt(hdim)
```

语义：`s_host_ref[h, q, k] = (Q_dequant[h,q,:] · K_dequant[h,k,:]) * scale_s`

注意这里用的是**反量化后的 Q/K**，而不是原始 float Q/K，因此这一步引入了 MXFP4 量化误差。这也是最终验证需要宽松容忍度（`atol=0.26`）的主要原因。

---

### Step 4：delta_s 修正（SA3 专属）

这是 SA3 与普通 MX 路径唯一的分叉点：

```cpp
if constexpr(is_sageattnv3)
{
    // s_host_ref：[nhead, real_seqlen_q, real_seqlen_k]
    // delta_s_host：[batch, nhead, num_q_blocks, seqlen_k]  (flat array)
    // kM0_sa3 = 64，与 GPU pipeline Q tile 大小对齐

    const index_t ds_batch_offset = b_idx * nhead * num_q_blocks_sa3 * shape_seqlen_k;

    s_host_ref.ForEach([&](auto& self, auto idx) {
        const index_t h_idx  = idx[0];   // head
        const index_t q_idx  = idx[1];   // query token（行内绝对位置）
        const index_t k_idx  = idx[2];   // key token
        const index_t qi_blk = q_idx / kM0_sa3;  // 哪个 Q-block（64行为一组）

        const index_t ds_off = ds_batch_offset
                             + h_idx * num_q_blocks_sa3 * shape_seqlen_k
                             + qi_blk * shape_seqlen_k
                             + key_offset + k_idx;  // key_offset 处理 split-kv

        self(idx) += static_cast<SMPLComputeDataType>(
            delta_s_host.data()[ds_off] * scale_s_host);
    });
}
```

**这段代码的语义**：

对 `s_host_ref` 的每个元素 `[h, q, k]`，加上 `delta_s[b, h, q/kM0, k] * scale_s`。

等价于：`S[h, q, k] += (q_mean[b, h, q/kM0] · K[b, h, k, :]) * scale_s`

这正是 SA3 算法中的修正项——把因均值平滑产生的"缺失项"补回来。修正后的 `s_host_ref` 在数学上等价于 `(Q_smooth_dequant @ K_smooth_dequant^T + delta_s) * scale_s`，即 GPU kernel 计算的完整注意力分数。

**关于 delta_s 的来源**：在 example runner 中，`delta_s_host` 的数据不是用 CPU 从 Q/K 计算出来的，而是**用随机值直接填充**的（`FillUniformDistribution<float>{-0.5f, 0.5f}`），同一份数据通过 `delta_s_buf.ToDevice()` 同时传给 GPU kernel 和用于 CPU 验证。这样做的目的是解耦：验证的是 GPU kernel 在给定 delta_s 下能否正确加回修正项，而不验证 delta_s 本身的计算正确性（那是 preprocess kernel 和 `test_fmha_sageattn_preprocess` 测试的职责）。

**index offset 细节**：`key_offset` 是当前 batch 在 K 序列上的起始偏移（batch mode 下为 0，split-kv 模式下非零）。`qi_blk * shape_seqlen_k` 而不是 `qi_blk * real_seqlen_k`，因为 delta_s 始终按全量 `shape_seqlen_k` 分配，即使实际有效长度 `real_seqlen_k` 更短。

---

### Step 5：masking（所有类型共用）

SA3 不修改这部分。causal mask、window mask、no_mask 分支和其他数据类型完全一样，都在 `s_host_ref` 上就地操作，将被 mask 掉的位置设为 `-inf`。

---

### Step 6：softmax（host P 保持 float）

```cpp
// PDataType 在 host 端被强制覆盖为 float（is_mx=true 路径）
// 原始 TypeConfig::PDataType = pk_fp4_t，但 host 不量化 P
reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
    s_host_ref, p_host_ref, p_compute_element_func, lse_host_ref);
```

SA3 的 `p_compute_element_func` 是 `identity{}`（`scale_p_host = 1.0f`），没有额外缩放。

**为什么 host 端 P 不量化为 MXFP4？**

注释明确说明了原因：

> P is not quantized and then dequantized here (PDataType = float).
> On host softmax is computed for the whole row of S, while on device FA computes
> softmax and quantizes it in **blocks of N0 values**.
> Quantization on host would make reference results **less** precise than the device results
> for large seqlen_k!

GPU kernel 的 online softmax 是按 N0（kN0=64）块更新的，每个 N0 块的 P 值用本块的局部 scale 量化；而如果 host 用全行 softmax 后的 P 统一量化，scale 会更大（覆盖全行最大值），反而会引入比 GPU 更大的量化误差，导致参考结果比 GPU 结果更不准确，使比较失去意义。因此 host 端干脆不量化 P，直接用 float 做 GEMM1。

---

### Step 7：V 反量化

```cpp
// v_host_ref 布局：[nhead, hdim_v, seqlen_k]（已是 column-major 转置后的形式）
// v_descale_host_ref 布局：[nhead, hdim_v, seqlen_k/32]

auto v_host_ref2 = reference_batched_mx_descale<
    VDataType,    // pk_fp4_t
    VScaleDataType, // e8m0_t
    OaccDataType, // float
    OaccDataType  // float
>(v_host_ref, v_descale_host_ref, kVScaleGranularity /*=32*/);
```

注意 `v_host_ref` 的 shape 是 `[nhead, hdim_v, real_seqlen_k]`，即转置后的布局——每行是一个 hdim channel 的所有 seqlen_k 个值，scale 粒度是沿 seqlen_k 方向每 32 个。这和 Q/K 的 scale 粒度（沿 hdim 方向每 32 个）是对称的。

---

### Step 8：PV GEMM（计算输出 O）

```cpp
reference_batched_gemm<
    PDataType,    // float（host 端 P，未量化）
    OaccDataType, // float（反量化后的 V）
    OaccDataType, // float
    ODataType     // float（输出）
>(p_host_ref, v_host_ref2, o_host_ref,
  identity{}, identity{},
  oacc_element_func);  // identity{}（SA3 无输出缩放）
```

`o_host_ref` 形状 `[nhead, real_seqlen_q, hdim_v]`，元素类型 float。

这一步等价于：`O_ref[h, q, :] = Σ_k P_float[h, q, k] * V_dequant[h, :, k]`

（V 的下标顺序注意：v_host_ref2 形状是 `[nhead, hdim_v, seqlen_k]`，GEMM 把 seqlen_k 作为收缩维，输出 `[nhead, seqlen_q, hdim_v]`。）

---

### Step 9：数值对比

```cpp
auto [rtol, atol] = get_elimit<FmhaFwdSageAttnV3>(init_method);
// rtol = 0.1, atol = 0.26

check_err(o_host_result, o_host_ref, "OUT Error: Incorrect results!", rtol, atol);
// 判定标准：|gpu[i] - cpu[i]| <= atol + rtol * |cpu[i]|
```

**为什么容忍度这么宽（rtol=0.1, atol=0.26）？**

- Q/K/V 都是 MXFP4 量化（4 位）精度，每次 descale 引入 `O(1/16)` 量级的相对误差
- GEMM0（`Q_dequant @ K_dequant^T`）的累加误差与 hdim 成正比（hdim=128）
- delta_s 的加法误差量级与 scale_s 相同
- softmax 的指数运算对输入误差有放大效应
- GEMM1（`P @ V_dequant`）又一次引入量化误差

这些误差逐级叠加，最终输出误差的典型量级在 0.1~0.2 范围内，与 `FmhaFwdMxFp4` 的容忍度（`rtol=0.1, atol=0.26`）完全相同——SA3 和普通 MX 的量化位宽一致，所以精度期望也一致。

---

### 验证路径与 GPU pipeline 的完整对应关系

| CPU reference 步骤 | 使用的函数 | GPU kernel 对应操作 |
|---|---|---|
| Q mx_descale | `reference_batched_mx_descale` | MFMA 读 Q_hat 时隐式 descale（硬件） |
| K mx_descale | `reference_batched_mx_descale` | MFMA 读 K_hat 时隐式 descale（硬件） |
| S = QK^T * scale_s | `reference_batched_gemm` | `BlockGemmMxARegBSmemCReg`（MXFP4 MFMA） |
| S += delta_s * scale_s | `s_host_ref.ForEach` | pipeline 中 `s_acc(i_j_idx) += delta_s_tile(i_j_idx)` |
| masking | `reference_batched_masking` | pipeline 中 mask 判断分支 |
| softmax(S) → P_float | `reference_batched_softmax` | pipeline online softmax + MXFP4 量化 P |
| V mx_descale | `reference_batched_mx_descale` | MFMA 读 V_hat 时隐式 descale（硬件） |
| O = P @ V | `reference_batched_gemm` | `BlockGemmMxARegBSmemCReg`（P 两级量化后的 MXFP4 MFMA） |
| check_err | — | — |

---

## 12. 单元测试：`test_fmha_sageattn_preprocess`（预处理 CPU 正确性）

### 文件位置

```
test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp
```

### 整体结构

测试用 GTest 的参数化测试框架：

```cpp
class SageAttnPreprocessTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>>
// 参数：(B, H, seqlen_q, seqlen_k, hdim)
```

测试用例通过 `INSTANTIATE_TEST_SUITE_P` 注册三组参数：

```cpp
INSTANTIATE_TEST_SUITE_P(SmallShapes, SageAttnPreprocessTest, ::testing::Values(
    std::make_tuple(1, 1,  64,  64,  64),  // 最小配置
    std::make_tuple(2, 8, 128, 256, 128),  // 中等配置（多 batch 多 head）
    std::make_tuple(1, 4, 256, 512,  64)   // 较长 seqlen
));
// 共 3 组参数 × 4 个 TEST_P = 12 个测试用例
```

编译期守卫：

```cpp
#ifdef CK_USE_NATIVE_MX_SUPPORT
    // 实际测试逻辑
#else
    TEST(SageAttnPreprocessTest, SkippedOnNonGfx950) {
        GTEST_SKIP() << "requires gfx950";
    }
#endif
```

`CK_USE_NATIVE_MX_SUPPORT` 只在 `-DGPU_TARGETS="gfx950"` 时被定义，确保 SA3 测试不会在不支持 MX 硬件的机器上编译失败或产生误导性结果。

---

### 辅助函数

```cpp
// 生成 [-1, 1] 均匀分布的随机 float32 数组，固定种子=42（可复现）
static std::vector<float> RandFloat(int n, float lo=-1.f, float hi=1.f);

// 元素级比对，使用混合误差标准
static bool AllClose(const std::vector<float>& a,
                     const std::vector<float>& b,
                     float atol=1e-4f, float rtol=1e-3f);
// 判定标准：|a[i] - b[i]| <= atol + rtol * |b[i]|
// atol 处理绝对误差（接近 0 的值）
// rtol 处理相对误差（数值较大的值）
```

---

### 四个测试用例详解

#### `TEST_P(SageAttnPreprocessTest, QMode)`

**验证目标**：`reference_sageattn_q_smooth` 函数的输出正确性。

**测试策略**：调用 CPU reference，然后对结果做**独立数学验证**（不是和另一个 reference 比，而是自己重算）：

```cpp
// 验证 q_mean：手动对每个 (b=0, h=0) 的 block 和 channel 重算均值，和 reference 比较
for qi in 0..num_q_blocks:
    for d in 0..4:  // 只检查前 4 个 channel（速度/覆盖的 trade-off）
        expected_mean = mean(Q[0,0, qi*kM0:(qi+1)*kM0, d])
        EXPECT_NEAR(q_mean_ref[0,0,qi,d], expected_mean, 1e-5f)

// 验证 q_smooth：对每个元素检查 q_smooth = Q - q_mean
for n in 0..4:
    qi = n / kM0
    for d in 0..4:
        expected = Q[0,0,n,d] - q_mean_ref[0,0,qi,d]
        EXPECT_NEAR(q_smooth_ref[0,0,n,d], expected, 1e-5f)
```

注意只抽查前 4 个 channel 和前 4 个 token，是为了测试速度（全量验证 O(seqlen×hdim) 很慢），但覆盖了边界条件（第 0 个 Q block）。

#### `TEST_P(SageAttnPreprocessTest, KMode)`

**验证目标**：`reference_sageattn_k_smooth` 的全局均值计算正确性。

```cpp
// 验证 k_mean：手动对每个 channel 累加所有 (b,h,n) 的值并除以总数
for d in 0..4:
    expected_mean = Σ_{b,h,n} K[b,h,n,d] / (B*H*seqlen_k)
    EXPECT_NEAR(k_mean_ref[d], expected_mean, 1e-4f)
    // 注意：atol 放宽到 1e-4，因为大量累加会产生浮点累积误差

// 验证 k_smooth = K - k_mean（广播）
for d,b=0,h=0,n in 0..4:
    expected = K[0,0,n,d] - k_mean_ref[d]
    EXPECT_NEAR(k_smooth_ref[0,0,n,d], expected, 1e-5f)
```

K 均值的验证比 Q 的 atol 宽松一个量级（1e-4 vs 1e-5），因为跨 `B*H*seqlen_k` 累加会积累浮点舍入误差。

#### `TEST_P(SageAttnPreprocessTest, VMode)`

**验证目标**：`reference_sageattn_v_transpose` 的转置正确性。

```cpp
// 直接验证下标交换语义
for b=0,h=0, n in 0..4, d in 0..4:
    src       = V[0, 0, n, d]           // 原始 [seqlen_k, hdim_v] 存储
    transposed = V_T_ref[0, 0, d, n]   // 转置后 [hdim_v, seqlen_k] 存储
    EXPECT_EQ(transposed, src)          // 精确相等（纯内存重排，无计算）
```

转置是纯内存重排，没有浮点运算，所以用 `EXPECT_EQ` 而不是 `EXPECT_NEAR`。

#### `TEST_P(SageAttnPreprocessTest, DeltaS)`

**验证目标**：`reference_sageattn_delta_s` 的矩阵乘法正确性。

```cpp
// 随机生成 q_mean 和 K，计算 delta_s，再手动点积验证
for b=0,h=0, qi in 0..2, kj in 0..4:
    expected = Σ_d q_mean[0,0,qi,d] * K[0,0,kj,d]   // 手动点积
    EXPECT_NEAR(delta_s[0,0,qi,kj], expected, 1e-4f)
```

为什么抽查 `qi < 2, kj < 4`：对 `hdim=128` 的矩阵乘，每次点积要做 128 次 MAC，选少数几个验证条目足以确认实现正确，同时避免测试本身变成慢 O(n²) 操作。

---

### 测试的特殊设计说明

**为什么没有端到端 GPU 验证**：

```cpp
// GPU kernel invocation is omitted in this test because SageAttnPreprocessKernel
// requires HIP device memory and cannot be launched without hardware access.
```

这批测试的定位是**纯 CPU 数学正确性测试**，验证 CPU reference 函数本身的语义正确。GPU kernel 的正确性由 `test_fmha_fwd_sageattnv3` 通过 E2E 输出对比验证（见第 11 节）。

这种**分离验证策略**的好处是：预处理 CPU reference 的正确性和 FMHA kernel 的正确性分开验证，两者不互相依赖，定位 bug 更清晰。

---

## 13. E2E Bug 修复（第二个 commit）

第二个 commit `aede18aaf9` 修复了跑 E2E 测试时发现的 5 个 bug：

### Bug 1：bit_cast 大小不匹配

**文件**：`sageattn_preprocess_pipeline.hpp`

**根因**：`bit_cast<X>(Y)` 要求 `sizeof(X) == sizeof(Y)`。`uint32_t >> 23` 结果还是 `uint32_t`（4 字节），无法 bit_cast 到 `uint8_t`（1 字节）。

```cpp
// 错误：编译期 static_assert 失败
dst_scale_ptr[g] = bit_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

// 正确：先移位（结果仍是 uint32_t），再 static_cast 截断到 8 位
dst_scale_ptr[g] = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);
```

e8m0 编码存的是 float 的指数字节（bits [30:23]，共 8 位），移位后取低 8 位就是正确的 e8m0 值。

### Bug 2：quant scale 字符串不识别

**文件**：`quant.hpp`

Example binary 通过 `-qscale=sageattnv3` 传字符串，但 decode 函数只认 `"sav3"`，导致运行时抛 `invalid_argument`。修复见第 9 节。

### Bug 3：is_mx guard 误拦 SA3

**文件**：`fmha_fwd_runner.hpp`

`is_mx = is_any_of<DataType, FmhaFwdMxFp8, FmhaFwdMxFp4, FmhaFwdSageAttnV3>` — SA3 被算进了 MX 家族。原来的检查：

```cpp
if(is_mx && qscale.type != quant_scale_enum::mx)
    // SA3 走到这里：is_mx=true, qscale=sageattnv3 ≠ mx → 报错
    return fwd_result::invalid_args;
```

修复：在条件中排除 SA3：

```cpp
if(is_mx && !is_sageattnv3 && qscale.type != quant_scale_enum::mx)
    return fwd_result::invalid_args;
```

### Bug 4：group 模式返回错误码

**文件**：`fmha_fwd_runner.hpp`

原来返回 `fwd_result::invalid_args`（值为 2），`CHECK_RESULT` 宏只跳过 `no_instance`（值为 1），所以 group 模式的测试变成了 FAILED 而不是 SKIPPED。修复：改为返回 `no_instance`。

### Bug 5：delta_s_ptr 未被赋值（空指针 crash）

**文件**：`fmha_fwd.hpp`

`MakeKargsImpl` 是一个通用函数，签名里没有 `delta_s_ptr` 参数，所以 SA3 专属字段调用后仍是默认值 `nullptr`。

kernel 启动后访问 `kargs.delta_s_ptr` 时触发 GPU 非法访问错误：
```
Aborted (core dumped) / GPU fault address (nil)
```

修复：在 `fmha_fwd.hpp` 的 `fmha_fwd_create_kargs_and_grids` 函数中，`MakeKargsImpl` 调用之后补充 SA3 字段（见第 7 节）。

---

## 14. 死代码清理（第三个 commit）

第三个 commit `b3c2f40f58` 清理了实现过程中遗留的死代码，共删除 143 行：

| 删除内容 | 文件 | 原因 |
|---------|------|------|
| `reference_sageattn_fwd()`（86行） | `reference_sageattn_preprocess.hpp` | 从未被任何代码调用；example runner 用多步骤 mx_descale + GEMM + delta_s 验证，不使用此函数 |
| `AllClose()` 静态方法（19行） | `test_fmha_sageattn_preprocess.cpp` | 测试用例全用 `EXPECT_NEAR`/`EXPECT_EQ`，此方法从未被调用 |
| `schedule_gemm_0` lambda + `DS_READ`/`MFMA` 常量（29行） | `block_fmha_pipeline_qr_ks_vs_sageattn.hpp` | lambda 标注了 `[[maybe_unused]]` 且从未调用；两个常量仅在此 lambda 中使用 |
| 重复的 if/else 分支（4行） | `block_fmha_pipeline_qr_ks_vs_sageattn.hpp` | num_total_loop ≤ 0 的 LSE early-exit 路径中 if/else 两分支代码完全相同，合并为单条语句 |

---

## 15. 关键工程细节汇总

| 技术点 | 描述 | 位置 |
|--------|------|------|
| stride-0 广播 | `make_tuple(number<0>{}, number<1>{})` 使 [1, seqlen_k] 数据像 [kM0, seqlen_k] 一样被 tile distribution 访问 | `fmha_fwd_kernel.hpp:2215` |
| `operator()` vs `operator[]` | 分布式张量的 `operator[]` 是 const（只读），`operator()` 是写访问；累加时必须用 `s_acc(i_j_idx) +=` | pipeline.hpp 中所有累加操作 |
| p_scale_factor = 6.0 的由来 | FP4 E2M1 格式的最大可表示值恰好是 6.0，用它填满动态范围使量化误差最小 | pipeline Step 4 |
| kQLoadOnce = true | Q 在循环前一次性加载到寄存器，所有 KV tile 迭代复用，减少全局内存访问 | pipeline.hpp:48 |
| e8m0 编码原理 | e8m0 只存 float 指数部分（bits[30:23]），表示 `2^(exp-127)`，无尾数精度损失 | preprocess_pipeline.hpp |
| host P 不量化（is_mx 路径） | `PDataType = std::conditional_t<is_mx, float, ...>`，host 端 softmax 输出保持 float，避免全行量化比 GPU 块量化误差更大 | fmha_fwd_runner.hpp:275 |
| delta_s 随机填充而非计算 | example runner 用随机值填充 delta_s 并同时传给 GPU 和 CPU，解耦 FMHA kernel 验证与 preprocess kernel 验证 | fmha_fwd_runner.hpp:984 |
| VERBATIM 关键字 | 阻止 ninja 对 CMake 生成命令中通配符进行 shell 展开 | CMakeLists.txt |
| `is_mx` 包含 SA3 | `is_any_of<..., FmhaFwdSageAttnV3>` 中 SA3 被纳入 MX 家族，所有 MX 的通用检查需要显式排除 SA3 | fmha_fwd_runner.hpp |

---

*文档更新时间：2026-03-06，对应 commit range `ce98eb4c..b3c2f40f`*
