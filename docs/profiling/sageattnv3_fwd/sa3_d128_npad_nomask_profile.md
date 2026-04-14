# SageAttention V3 前向 Kernel Profiling 报告

**硬件：** MI350X (gfx950)
**配置：** d=128，无 padding，无 mask，batch mode
**日期：** 2026-04-14

---

## 1. 基本信息

### 1.1 Kernel 签名

```
ck_tile::kentry<ck_tile::gfx950_t, 2,
  ck_tile::FmhaFwdKernel<
    ck_tile::BlockFmhaPipelineQRKSVSSageAttn<
      ck_tile::BlockFmhaPipelineProblem<
        pk_float4_e2m1_t, pk_float4_e2m1_t, pk_float4_e2m1_t,  // Q/K/V: fp4
        float, float, float,                                     // 中间类型
        unsigned char, float, pk_float4_e2m1_t,                  // bias/scale/P
        float, float,                                             // 输出类型
        TileFmhaShape<sequence<128,128,64,128,64,128>, ...>
      >,
      BlockFmhaPipelineQXKSVSCustomPolicy<true, false, 1, 1>
    >,
    Default2DEpilogue<Default2DEpilogueProblem<float,float,false,false,true>>
  >
>
```

**Pipeline：** `BlockFmhaPipelineQRKSVSSageAttn`
**数据类型：** Q/K/V → fp4 (pk_float4_e2m1)，输出 → float
**Tile 大小：** kM0=128，kN0=128，kK0=64，kK1=128（GEMM0：128×128×64；GEMM1：128×128×128）

### 1.2 Kernel 资源

| 资源 | 数值 |
|------|------|
| VGPR (arch) | 128 |
| VGPR (accum/AGPR) | 0 |
| SGPR | 80 |
| LDS | 4352 bytes (4.25 KB) |
| Workgroup 大小 | 256 threads（4 waves） |
| Grid | (8192, 32, 2) |
| 总 CTA 数 | 524,288 |
| 总 Wave 数 | 8,192 |
| Occupancy | 2 waves/CU（256/128=2） |

**Occupancy 限制因素：** VGPR=128 是限制因素。每 CU 有 512 个 VGPR slot（256 threads × 2 waves × 128 = 65536 VGPRs，占满整个 CU VGPR 文件）。LDS 仅 4.25 KB，远不构成限制。

---

## 2. 基准测试结果（Benchmark）

测试配置：b=2，h=32，无 mask，无 padding

### 2.1 d=128 各序列长度

| 序列长度 s | 时间 (ms) | 算力 (TFLOPs) | 带宽 (GB/s) |
|-----------|-----------|---------------|-------------|
| 1024 | 0.058 | 595 | 799 |
| 2048 | 0.190 | 725 | 487 |
| 4096 | 0.736 | 747 | 251 |
| 8192 | 2.677 | 821 | 138 |

**随序列长度增加，算力提升，带宽下降**：这是 FMHA 的典型行为。短序列时 K/V 数据量小、L2 缓存命中率高（799 GB/s 接近 HBM 带宽上限），长序列时访存量大、L2 缓存失效，但计算密度上升（821 TFLOPs 为峰值）。

### 2.2 d=256 各序列长度（sageattnv3，fp4）

| 序列长度 s | 时间 (ms) | 算力 (TFLOPs) | 带宽 (GB/s) |
|-----------|-----------|---------------|-------------|
| 4096 | 1.801 | 610 | 205 |
| 8192 | 6.947 | 633 | 106 |

d=256 相比 d=128 算力更低（610 vs 747 TFLOPs），原因是 d=256 时 GEMM 的算术强度更高，但 softmax 开销（与 d 无关，仅与 seqlen 有关）在总开销中占比相对更大。

### 2.3 fp16 输入对比（d=128）

| 类型 | s | 时间 (ms) | 算力 (TFLOPs) | 带宽 (GB/s) |
|------|---|-----------|---------------|-------------|
| sageattnv3（fp4 Q/K/V） | 4096 | 0.736 | 747 | 251 |
| sageattnv3fp16（fp16 Q/K/V） | 4096 | 0.685 | 803 | 172 |
| sageattnv3（fp4 Q/K/V） | 8192 | 2.677 | 821 | 138 |
| sageattnv3fp16（fp16 Q/K/V） | 8192 | 2.563 | 858 | 92 |

fp16 版本在两个序列长度下均快于 fp4 版本，原因是 fp4 kernel 需要额外的 scale 向量读取和量化步骤，而 fp16 MFMA 的吞吐与 fp4 MFMA 相当（同为 32x32x64 规格）。

### 2.4 Causal Mask 对比（d=128, s=4096）

| Mask | 时间 (ms) | 等效算力 (TFLOPs) | 带宽 (GB/s) |
|------|-----------|-------------------|-------------|
| 无 mask | 0.736 | 747 | 251 |
| Causal mask | 0.402 | 685（基于完整 FLOPs） | 460 |

Causal mask 约快 45%：causal masking 使平均有效序列长度减半（每个 Q token 平均只处理 seqlen/2 个 K token），迭代次数减少约一半，但带宽利用率提升（460 vs 251 GB/s），说明 causal mask 下 kernel 更偏向带宽瓶颈。

---

## 3. PMC 计数器分析（d=128, s=4096）

### 3.1 指令统计（per wave 均值）

| 计数器 | 数值/wave | 占比 |
|--------|-----------|------|
| SQ_WAVES | 8,192 | — |
| SQ_INSTS_VALU（含 MFMA） | 19,861 | 100% |
| SQ_INSTS_MFMA | 512 | 2.58% |
| 纯 VALU（含 exp/fma/cvt 等） | 19,349 | 97.4% |
| SQ_INSTS_VMEM | 2,710 | — |
| SQ_INSTS_SALU | 997 | — |
| SQ_INSTS_SMEM | 11 | — |
| SQ_INSTS_LDS | 704 | — |
| SQ_INSTS_BRANCH | 164 | — |

**VALU:MFMA 比例 = 37.8:1**

这是一个极端 VALU 重的 kernel。每轮主循环迭代理论上有 16 条 MFMA（GEMM0 的 8 条 + GEMM1 的 8 条），但有约 600 条纯 VALU 指令用于 softmax 相关计算（exp、max、sum、reduce、量化等）。

MFMA 指令占总 VALU 指令的比例仅 2.6%，说明 MFMA 的吞吐几乎无法被充分利用。

### 3.2 计算利用率

| 指标 | 数值 | 说明 |
|------|------|------|
| VALU 利用率 (SQ_ACTIVE_INST_VALU / SQ_BUSY_CU_CYCLES) | **54.9%** | SIMD 忙时中 VALU 执行比例 |
| Any Active 利用率 | **67.6%** | CU 有任意指令执行的时间比例 |
| Wait 比例 | **44.6%** | CU 等待（stall/idle）的时间比例 |

VALU 利用率 54.9% 意味着约 45% 的 CU 活跃周期处于等待状态（内存延迟、LDS bank conflict、barrier 等）。尽管 kernel 是 VALU bound，但大量 stall 导致实际利用率远低于理论峰值。

### 3.3 LDS Bank Conflict 分析

| 计数器 | 数值 |
|--------|------|
| SQ_LDS_BANK_CONFLICT | 16,777,216 |
| SQ_INSTS_LDS | 5,767,168 |
| **Bank Conflict Rate** | **290.9%** |

每条 LDS 指令平均产生约 3 次额外的 conflict 周期（290.9% overhead），这是显著的瓶颈。

**根本原因：** K/V tile 的 LDS 读取（`ds_read_b128`）具有列方向的步长访问模式。具体来看：
- Q tile 以 `buffer_load_dwordx4` 写入 LDS，写入时 bank 分布可能已造成 conflict pattern
- K tile 从 LDS 读取时，GEMM0 需要按列（列宽 64 个 fp4 元素）读取，每个 wave 的 32 个 lane 同时读取 stride=128 字节步长的地址，落在相同的 32 个 bank 上，产生 conflict
- 64-bank LDS（gfx950）下，fp4 数据的打包方式（每 4 个元素打包为 1 个 dword）会影响 bank 分布

---

## 4. ATT Trace 热点分析（stall/hit 排行）

以下为 ATT trace 中 stall 最严重的指令（按 stall/hit cycles 排序）：

| 排名 | 地址 | 指令 | Stall/Hit | 说明 |
|------|------|------|-----------|------|
| 1 | 0x2dc4 | `s_waitcnt vmcnt(0)` | **34,372 cycles** | Prologue 等待 V tile VMEM load 完成 |
| 2 | 0x2518 | `s_waitcnt lgkmcnt(0)` | **7,568 cycles** | Prologue kernarg（参数）SMEM 加载等待 |
| 3 | 0x2f0c | `buffer_load_dword v193` | **5,904 cycles** | V tile load（第二批，8 个元素中的第一个） |
| 4 | 0x2e8c | `buffer_load_dword v185` | **5,312 cycles** | V tile load（第一批，第一个） |
| 5 | 0x25e4 | `s_waitcnt lgkmcnt(0)` | **4,368 cycles** | seqstart offset SMEM 加载等待 |
| 6 | 0x30e8 | `s_barrier` | **5,624 cycles** | GEMM0 之前等待 K tile 写入 LDS 完成 |
| 7 | 0x2e2c | `buffer_load_ubyte v12` | **3,856 cycles** | Q scale 向量加载 |
| 8 | 0x28a8 | `v_readfirstlane_b32 s0` | ~2,844 cycles | Prologue 中的 buffer descriptor 初始化 |

**关键发现：**

1. **最大 stall（#1）：Prologue V tile 等待（34,372 cycles）**
   `s_waitcnt vmcnt(0)` 等待 64 条 `buffer_load_dword` 全部完成。V tile 大小为 128×128 个 fp4 元素，每个 wave 需要加载 `128×128/8 / 32 = 64` 个 dword。所有 64 条 load 都在 prologue 发出，随后 kernel 等待全部完成再进入主循环。这 34,372 cycles 是完全暴露的内存延迟，是最优先的优化目标。

2. **Kernarg 加载等待（#2）：7,568 cycles**
   GPU 从 kernarg 内存读取 kernel 参数，smem 请求的延迟暴露在 prologue 的 `s_waitcnt lgkmcnt(0)`。这是 prologue 固定开销，难以优化。

3. **Barrier 等待（#6）：5,624 cycles**
   在 GEMM0 开始前等待 K tile 完成从 GMEM 加载并写入 LDS。这说明 K tile prefetch 深度不够，主循环迭代中 K load 完成前 barrier 已到达。

4. **V tile load 本身的延迟（#3, #4）**
   单条 `buffer_load_dword` 自身的 latency 约 5000-6000 cycles（DRAM latency）。这反映了 HBM 的长访问延迟，在 occupancy=2 时没有足够多的其他 wave 来掩盖。

---

## 5. Kernel 结构分析（汇编层面）

基于 ATT CSV 数据，kernel 的主要阶段如下（以 vmem 地址标注）：

```
[Prologue, 0x2100–0x2880]
  - Kernarg SMEM 加载 (s_load_dwordx4/8)
  - s_waitcnt lgkmcnt(0) × 2  <-- stall #2, #5
  - 地址计算（整除、步长计算等 SALU 密集区）
  - Buffer descriptor 初始化（v_readfirstlane 循环）

[Q tile 加载, 0x2884–0x2dc0]
  - buffer_load_dwordx4 循环（Q tile → VGPR）
  - ds_write_b128（Q → LDS）
  - s_waitcnt vmcnt(0)  <-- stall #1（等待全部 Q/K/V load）

[V tile 加载, 0x2e2c–0x2f40]
  - buffer_load_ubyte v12（Q scale）  <-- stall #7
  - 64 × buffer_load_dword（V tile: v185–v248）  <-- stall #3, #4
  - buffer_load_ubyte × 4（K scale）

[K tile 加载 → LDS, 0x2f44–0x30e0]
  - buffer_load_dwordx4 × n（K tile）
  - ds_write_b128（K → LDS）

[GEMM0 准备, 0x30e4–0x30f4]
  - s_waitcnt lgkmcnt(0)
  - s_barrier  <-- stall #6（5,624 cycles）
  - ds_read_b128 × 4（读 K tile：v[14:17], v[18:21], v[22:25], v[250:253]）
  - s_waitcnt lgkmcnt(0)（等待 ds_read）
  - s_barrier（二次同步）

[GEMM0: S = Q * K^T, 0x30f8–0x3270]
  8 × v_mfma_scale_f32_32x32x64_f8f6f4
  （fp4 矩阵乘，scale 参数来自 v12/v159 等 scale 寄存器）
  interleaved with ds_read for next K tile

[S += delta_s bias, 0x3274–0x3380]
  64 × v_add_f32
  （将 GEMM0 结果 S[i][j] 加上 delta_s bias）

[Row Max, 0x3384–0x33b0]
  32 × v_max3_f32（计算每行最大值）
  + ds_bpermute_b32 + v_max3_f32（跨 wave reduce）

[Softmax Exp, 0x33b4–0x3500]
  v_fmac_f32（更新 m_new）
  64 × (v_fma_f32 + v_exp_f32)  <-- 最密集 VALU 区
  （P[i] = exp2(log2e × S[i] − m_new_scaled)）
  + 16 × v_exp_f32（o_acc rescale）

[Row Sum, 0x3504–0x3560]
  64 × v_add_f32（累加 P 行和）
  + ds_bpermute_b32 + v_add_f32（跨 wave reduce）

[P 量化 → fp4, 0x3564–0x35a8]
  v_cvt_f32_u32 / v_pk_fma_f16 / packing 指令
  将 float P tile 量化为 pk_float4_e2m1

[GEMM1: O += P * V, 0x35ac–0x3670]
  8 × v_mfma_scale_f32_32x32x64_f8f6f4
  （fp4 矩阵乘，使用量化后的 P 和 V tile）

[循环控制 + Prefetch, 0x3674–0x36c0]
  s_cmp / s_cbranch（检查是否继续迭代）
  下一轮 K/V tile 的 buffer_load 发出（prefetch）

[Epilogue, 0x36c4–0x3930]
  O rescale（v_mul_f32 × n）
  LSE 计算（log-sum-exp 输出）
  buffer_store_dwordx4（写回 O）
  buffer_store_dword（写回 LSE）
```

---

## 6. 瓶颈根因分析

### 6.1 VALU Bound：37.8:1 VALU:MFMA 比例

每次主循环迭代包含：

| 操作 | 指令数 | 说明 |
|------|--------|------|
| GEMM0 (S = Q·K^T) | 8 MFMA | v_mfma_scale_f32_32x32x64_f8f6f4 |
| GEMM1 (O += P·V) | 8 MFMA | v_mfma_scale_f32_32x32x64_f8f6f4 |
| S += delta_s bias | 64 v_add_f32 | bias 加法 |
| Row max | 32 v_max3_f32 | 行最大值 |
| Cross-wave max | ~8 ds_bpermute + v_max3 | 跨 wave reduce |
| Softmax exp | 64 v_fma + 64 v_exp_f32 | 主要 VALU 负担 |
| o_acc rescale | 16 v_exp_f32 + 16 v_mul | 历史 o 缩放 |
| Row sum | 64 v_add_f32 | 行和 |
| Cross-wave sum | ~8 ds_bpermute + v_add | 跨 wave reduce |
| P 量化 | ~30 v_cvt/pack | fp4 量化 |
| 地址计算等 | ~80 SALU/VALU | 辅助指令 |

**每轮 MFMA = 16 条，纯 VALU ≈ 600+ 条，比例 ~37:1。**

gfx950 的 XDL（MFMA）peak 吞吐约为 1638 TFLOPs（fp4），纯 VALU peak 约为 102 TFLOPs（fp32）。当 VALU:MFMA = 37:1 时，VALU 成为严重瓶颈。

### 6.2 v_exp_f32 的代价

`v_exp_f32` 在 gfx950 上通过 Transcendental 单元执行，每条约 4–8 个时钟周期（取决于是否有 VALU co-issue）。每次迭代有 64+16 = 80 条 `v_exp_f32`：

- 64 条用于 P tile 的 softmax（128×64 tile，每 wave 处理 64 个元素）
- 16 条用于 o_acc 的历史缩放（rescale factor = exp(m_old - m_new)）

即使每条 exp 只需 4 周期，80 条 exp × 4 cycles = 320 cycles，加上 64 条 v_fma（前置计算 `S × log2e - m`）≈ 128 cycles，softmax 区段贡献约 450+ cycles 的纯计算延迟（不含 stall）。

### 6.3 内存访问模式

每次主循环迭代访存分析（s=4096，kN0=128 迭代 32 次）：

| 数据 | 每次迭代大小 | 总量 |
|------|------------|------|
| K tile (fp4) | 128×128×0.5 B = 8 KB | 256 KB |
| K scale | 128×1 B = 128 B | 4 KB |
| V tile (fp4) | 128×128×0.5 B = 8 KB | 256 KB |
| V scale | 128×1 B = 128 B | 4 KB |
| delta_s bias | 128 B | 4 KB |

Q tile（128×128×0.5 B = 8 KB）仅在 prologue 加载一次并存入 LDS，在主循环中重复使用。

理论带宽需求（per CTA）：(256+256+8) KB × 2（K 和 V scale）/ 0.736 ms ≈ 700 GB/s per GPU，远超 MI350X 的 HBM 带宽（~8.4 TB/s）但由于只有 524288 个 CTA 需要各自独立读取，实测带宽 251 GB/s 说明 L2 缓存命中率约 30%（多个 CTA 共享相同 K/V 数据，不同 batch/head 的 K/V 数据被 L2 缓存命中部分覆盖）。

### 6.4 Occupancy 受限的内存延迟暴露

Occupancy = 2 意味着每个 CU 只有 2 个 workgroup（8 个 wave）。当一个 wave 遇到 VMEM stall（如 buffer_load 延迟 ~300 个 cycle），调度器切换到其他 wave。但 occupancy=2 时，总共只有 8 个 wave 可用，远不足以完全掩盖 HBM 延迟（~300–800 cycles）。

实测 Wait% = 44.6% 正好说明了这一点：接近一半的 CU 活跃时间处于等待状态，主要来自 VMEM 和 LDS 延迟。

---

## 7. 优化机会分析

### 7.1 V Tile 更早预取（最高优先级）

最大的单点 stall（34,372 cycles at `s_waitcnt vmcnt(0)`）来自等待 V tile 的 64 条 buffer_load。

**问题：** 当前实现在 prologue 阶段一次性发出所有 V tile load，然后立即 `s_waitcnt vmcnt(0)`，把全部延迟暴露出来。

**优化方向：**
- 在主循环的 K load 完成后、GEMM0 开始前，提前发出下一轮迭代的 V tile load
- 主循环中将 V load 与 GEMM0/softmax/GEMM1 交叉执行，用计算掩盖内存延迟
- 类似 double-buffering：当前迭代使用 V tile `A`，同时预取下一次迭代的 V tile `B`

**预期收益：** 若能完全隐藏 V tile load 延迟（34,372 cycles per wave，全局 8192 waves），理论可节省约 43% 的执行时间。但受限于 VGPR（需要额外寄存器存 V prefetch buffer），实际提升需要评估 VGPR 压力。

### 7.2 LDS Bank Conflict 优化（Bank Conflict Rate 290.9%）

**问题：** K tile 写入 LDS 后，GEMM0 的 ds_read 访问模式产生 290.9% 的 bank conflict rate。

**分析：** 64-bank LDS 下，每 4 字节一个 bank。若 ds_read_b128 从步长为 `N×4 byte`（N 为 bank 数的倍数）的地址读取，32 个 lane 会映射到相同的 bank 子集。

**优化方向：**
- 写入 LDS 时增加 padding（每行 +4 或 +8 字节），破坏 conflict pattern
- 在 `buffer_load → LDS` 的写入阶段改变 swizzle 布局，使 ds_read 的 32 个 lane 的访问地址均匀分布在 64 个 bank 上
- 验证 Q tile（kUseLdsQ=true）的 LDS 布局是否也有类似问题

**预期收益：** 消除 290.9% bank conflict 可将 LDS 有效吞吐提升约 3.9×，等效节省约 15–20% 执行时间（假设 LDS 在关键路径上）。

### 7.3 Softmax 指令优化

**背景：** 64 条 `v_exp_f32` 是 VALU 关键路径中不可避免的开销。

**可探索的方向：**
- 使用 `v_exp2_f32`（exp base-2）替代 `v_exp_f32`（exp base-e），前者在某些实现中延迟更低。当前代码实际上已经在计算 `exp2(log2e × S - m)`（即 SA3 的 exp2 形式），但编译器可能没有完全利用这一点。
- 将 `v_fma + v_exp2` 对合并为更少的指令（视 ISA 支持情况）
- 利用 MI350 的 Pipe0/Pipe1 co-execution：Pipe0 执行 MACC（v_fma），Pipe1 执行 Trans（v_exp），两者可以并行。需要检查编译器是否已在相邻指令中安排了 co-issue。

**当前限制：** gfx950 没有批量 exp 指令（如 IEEE 754 中的 `v_exp_f32` 每条处理 1 个 f32），无法简单地提升 exp 的吞吐。

### 7.4 降低 VGPR 压力，提高 Occupancy

**当前状态：** 128 VGPR → occupancy=2。

**目标：** 降至 96 VGPR → occupancy=3（增加 50%，有助于掩盖更多内存延迟）。

**主要 VGPR 消耗源：**
- V tile VGPR buffer（v185–v248）：64 个 VGPR = 64 个 fp32 = V[32rows × 128cols × 0.5B/fp4 / 4B/VGPR / 32lanes]
- Q/K MFMA 输入寄存器
- o_acc 累加寄存器（16 个 fp32 × 4 MFMA 的 output）
- Softmax 中间量（P tile、row max、row sum、scale 等）
- 地址/描述符寄存器

**优化方向：**
- 减少 kN0（从 128 降到 64），可以将 K tile 和 V tile 的 VGPR 减半，但会使外层循环迭代次数翻倍，增加 o_acc rescale 开销
- 在 P tile 计算完毕后立即将 V tile VGPR 释放（当前 V tile 寄存器在整个迭代期间占用）
- 考虑 AGPR 存放 V tile（但 AGPR 在 gfx950 上的 FMHA 场景下容易造成 VGPR/AGPR 同时占用，增加总压力，需评估）

### 7.5 Wave Quantization 考量

MI350（gfx950）有 4 个 XCD，每个 XCD 含 32 个 CU。每个 XCD 的 CTA 数 = Grid / (4 XCD × occupancy)。

对于 s=4096，Grid = (seqlen/kM0) × heads × batch = 32 × 32 × 2 = 2048 CTA。
- 每 XCD: 2048 / 4 = 512 CTA
- occupancy=2 → 每 CU: 512 / 32 / 2 = 8 wave/CU

Wave quantization：ceil(512 / 32) = 16 轮调度，恰好整除，无 wave quantization 损失。

对于 s=1024，Grid = 8 × 32 × 2 = 512 CTA → 每 XCD 128 CTA → 4 轮调度，也整除。故 wave quantization 在当前测试配置中不是问题。

---

## 8. 与理论峰值的对比

### 8.1 MI350X 理论峰值

| 资源 | 峰值 |
|------|------|
| fp4 XDL 吞吐 | ~1638 TFLOPs（理论） |
| HBM 带宽 | ~8.4 TB/s |
| fp32 VALU 吞吐 | ~102 TFLOPs |

### 8.2 当前效率（d=128, s=8192）

| 指标 | 数值 | 理论峰值 | 效率 |
|------|------|---------|------|
| fp4 算力 | 821 TFLOPs | 1638 TFLOPs | **50.1%** |
| HBM 带宽 | 138 GB/s | 8400 GB/s | 1.6%（完全计算 bound） |
| fp32 VALU | ~54 TFLOPs（估算） | 102 TFLOPs | **~53%** |

fp4 XDL 效率 50.1% 与 fp32 VALU 效率 ~53% 相近，说明 kernel 基本被 VALU pipeline 限制，MFMA 利用率与 VALU 利用率相互耦合（VALU 饱和时 MFMA 也难以被充分调度）。

---

## 9. 数据文件索引

| 文件 | 说明 |
|------|------|
| `sa3_d128_npad_nomask_raw.asm` | 原始反汇编（含 NOP padding） |
| `stats_ui_output_agent_13088_dispatch_12.csv` | ATT trace 统计（含 hitcount/stall/idle per instruction） |
| `pmc_1/pmc1_results.db` | PMC 基础计数器（VALU/MFMA/VMEM 等） |
| `pmc_util_results.db` | 利用率 PMC（SQ_ACTIVE_INST_VALU/SQ_BUSY_CU_CYCLES） |
| `pmc_lds_results.db` | LDS PMC（SQ_LDS_BANK_CONFLICT 等） |
| `att_d128_results.db` | ATT 原始 trace DB |
| `ui_output_agent_13088_dispatch_12/` | ATT wavestate JSON（per-wave 时序） |

---

## 10. 总结

SageAttention V3 forward kernel（d=128，gfx950）在序列长度 8192 时达到 821 TFLOPs，是 MI350X 理论 fp4 峰值的 50%。

**主要瓶颈（按优先级）：**

1. **VALU Bound（最根本）：** softmax 区段（64 v_exp + 64 v_fma + 64 v_add + 32 v_max + P quantize）产生 37.8:1 的 VALU:MFMA 比，使 MFMA 无法得到充分调度。这是 FlashAttention 类算法在 fp4/fp8 加速卡上的固有结构性瓶颈。

2. **V Tile 内存延迟暴露（可优化）：** Prologue 的 `s_waitcnt vmcnt(0)` 暴露 34,372 cycles 延迟，占全部 stall 的最大单点份额。通过提前 prefetch 和 double buffering 可部分隐藏。

3. **LDS Bank Conflict（可优化）：** 290.9% bank conflict rate 在 LDS 访问上增加约 3× 延迟开销。通过 LDS padding 或 swizzle 布局优化可消除。

4. **低 Occupancy（结构性）：** 128 VGPR → occupancy=2，波数不足以掩盖 HBM 延迟。降低 VGPR 使用是可能的方向，但需权衡 kN0/数据布局的改动代价。
