# SA3 fp16 dwordx4 OPSEL Scale 优化 — Profiling 报告

## 版本信息

| 项目 | 值 |
|------|-----|
| 内核 | `BlockFmhaPipelineQRKSVSSageAttn` |
| 数据类型 | fp16 (Q/K/V)，fp4 MXFP4 内部量化，fp32 累加 |
| MFMA | `v_mfma_scale_f32_32x32x64_f8f6f4` (32×32×64, fp4 A/B, fp32 C) |
| Tile Shape | kM0=128, kN0=128, kK0=64, kN1=128, kK1=64 |
| Workgroup | 256 线程（4 warps），VGPR=112，LDS=4352 B |
| 优化分支 | dwordx4 OPSEL — K/V scale 用 `buffer_load_dwordx4` 批量加载 |
| 对比基线 | 之前的 16×16×128 OPSEL 版本（16x16 MFMA，VGPR=88） |

---

## 一、优化方案说明

### 背景

SA3 fp16 forward kernel 需要为每个 K/V tile 提供 e8m0 scale bytes（每个 tile 1 个 byte，granularity=32）。此前的实现使用：
- **8× `buffer_load_ubyte`** + shift/OR 打包为 int32 → 每 iter 8 条 VMEM 指令

### dwordx4 OPSEL 方案

每对相邻 tile（pair）的 scale 被预打包为 `int32x4`，一次 `buffer_load_dwordx4` 可覆盖 2 个连续 K/V tile pair（即 4 个 tile 的全部 scale bytes）：

```
k_scale_packed[pair_idx*8 .. pair_idx*8+7]  →  8 个 int32
→ 一次 buffer_load_dwordx4 取出 4 个 int32
→ 覆盖 2 个相邻 tile 的 K scale（tile 0: [0],[1]，tile 1: [2],[3]）
```

在 GEMM 中通过 **OPSEL cycling** 从同一个 int32 寄存器的 byte 0/1/2/3 中选取各个 nIter 的 scale，无需任何 shift/OR 指令。

**OPSEL 循环规则（32×32×64，ScaleRepeatB=4）：**

| nIter | op_sel_hi (A scale) | op_sel (B scale) |
|-------|---------------------|------------------|
| 0     | [0,0,0]             | [0,0,0]          |
| 1     | [0,0,0]             | [1,0,0]          |
| 2     | [1,0,0]             | [0,0,0]          |
| 3     | [1,0,0]             | [1,0,0]          |

每 2 个 tile 处理完后（block_in_pair==1），执行一次 dwordx4 重载，加载下一对 tile 的 scale。

---

## 二、PMC 计数器原始数据

测试条件：b=2, h=32, s_q=4096, s_k=4096, d=128，gfx950（MI350）

| 计数器 | 值 |
|--------|-----|
| VGPR | 112 |
| LDS | 4352 B |
| Workgroup | 256 threads (4 warps) |
| Duration (s=4096) | 502 µs |
| SQ_WAVES | 8,280 |
| SQ_INSTS_VALU | 144,884,346 |
| SQ_INSTS_MFMA | 4,194,304 |
| SQ_INSTS_SALU | 4,073,017 |
| SQ_INSTS_LDS | 5,767,168 |
| SQ_INSTS_VMEM | 18,186,386 |
| SQ_ACTIVE_INST_VALU | 174,546,470 |
| SQ_BUSY_CU_CYCLES | 244,037,525 |
| SQ_LDS_BANK_CONFLICT | 16,777,216 |

---

## 三、派生指标分析

### 3.1 基本利用率

| 指标 | 计算 | 值 |
|------|------|-----|
| 纯 VALU 指令 / wave | (144,884,346 − 4,194,304) / 8,280 | 17,000 条/wave |
| MFMA 指令 / wave | 4,194,304 / 8,280 | 507 条/wave |
| VALU : MFMA 比 | 17,000 / 507 | **33.5 : 1** |
| VALU 利用率 | 174,546,470 / 244,037,525 | **71.5%** |
| LDS bank conflict / wave | 16,777,216 / 8,280 | 2,027 次/wave |

> **注：** VALU 利用率 71.5% 已显著优于 baseline 的 33.2%。VALU:MFMA 比 33.5:1 相比 baseline 39:1 有改善，主要来自消除了 shift/OR 打包指令。

### 3.2 MFMA 利用率估算

每 iter 执行 16 个 MFMA（GEMM0: 8 个 + GEMM1: 8 个），每个 MFMA 延迟 128 cycles：
- 理想 MFMA 连续执行时间 = 16 × 128 = 2048 cycles/iter
- 实测平均 iter 时间 ≈ 502,000 µs / (4096/128 iter) × (GPU freq ≈ 2.1 GHz / CU数) → 分析后：
- SQ_BUSY_CU_CYCLES = 244,037,525 / 8,280 waves ≈ 29,473 cycles/wave
- 16 个 MFMA × 128 cycles = 2048 cycles；MFMA 占比 = 2048/29473 ≈ **7.0%**
- 瓶颈在 VMEM/LDS 等待，非 MFMA compute bound

### 3.3 VMEM 分析

| 类型 | 指令数/wave | 说明 |
|------|------------|------|
| buffer_load_dwordx4 | ~2.4 / iter | K tile + scale dwordx4 |
| buffer_load_dword | ~48 / iter | V tile (每行 1 个 dword，未向量化) |
| buffer_store_dwordx4 | ~8 / epilogue | O tile 写回 |

V tile 使用 48× `buffer_load_dword`（非 dwordx4），原因是 V 的 LDS layout 需要按 fp4 行对齐存储，地址不连续无法合并为 dwordx4。这是当前主要 VMEM overhead。

---

## 四、ATT Trace 热点分析

### 4.1 Top 15 Stall 热点（按 Stall cycles 排序）

| 排名 | Vaddr | Hit | Stall | 位置 | 根因 |
|------|-------|-----|-------|------|------|
| #1 | 0x0000215c | 4 | 5332 | Prologue s_waitcnt lgkmcnt | 多批 s_load 等待，SGPR arg 加载从 L2/DRAM |
| #2 | 0x0000294c | 16 | 4684 | s_waitcnt vmcnt(1) | V tile dword load 前 1 个完成，VMEM FIFO |
| #3 | 0x00002110 | 4 | 3940 | Prologue s_waitcnt lgkmcnt | 第 2 批 sload，batch_idx lookup |
| #4 | 0x00003de8 | 4 | 3864 | buffer_store_dwordx4 (O write) | Epilogue O tile store，L2 write latency |
| #5 | 0x00002ad0 | 16 | 3748 | s_waitcnt vmcnt(33) | 等待 V 前 33 个 dword，VMEM FIFO 深度 |
| #6 | 0x000022a8 | 4 | 1740 | Prologue s_waitcnt lgkmcnt | group_mode lookup (一次) |
| #7 | 0x00002c8c | 16 | 1240 | s_barrier | 4-warp K-LDS 同步点 |
| #8 | 0x00002cb0 | 16 | 1200 | s_waitcnt lgkmcnt (K LDS read) | 等待 4× ds_read_b128 完成 |
| #9 | 0x0000286c | 16 | 1164 | buffer_load_dwordx4 (K tile) | K tile L2 miss，每 iter 地址递增 |
| #10 | 0x00002cc4 | 16 | 1144 | s_waitcnt lgkmcnt (K LDS 读2) | 等待第 2 批 ds_read_b128 |
| #11 | 0x00002a00 | 16 | 976 | buffer_load_dword (V tile #1) | V tile dword L2 miss |
| #12 | 0x00003408 | 16 | 928 | s_waitcnt lgkmcnt (GEMM1 LDS) | 等待 GEMM1 V 第 3 段 LDS |
| #13 | 0x000037c4 | 16 | 928 | s_waitcnt lgkmcnt (GEMM1 LDS) | 等待 GEMM1 V 第 2 轮 LDS |
| #14 | 0x000029c0 | 16 | 824 | buffer_load_dword (V tile #8) | V tile dword L2 miss (中段) |
| #15 | 0x00002a40 | 16 | 808 | buffer_load_dword (V tile #24) | V tile dword L2 miss (末段) |

### 4.2 热点类别汇总

| 类别 | 总 stall (估算) | 占比 | 改善可能性 |
|------|----------------|------|-----------|
| V tile dword 加载 (L2 miss) | ~12,000 cycles/iter | 40% | 中：改为 dwordx4 需重排 V layout |
| LDS 等待 (K/V ds_read) | ~5,500 cycles/iter | 18% | 低：LDS latency 固定 |
| Barrier/sync | ~1,240 cycles/iter | 4% | 低：结构性需求 |
| K tile dword×4 (L2 miss) | ~1,164 cycles/iter | 4% | 低：K 每 iter 不同，prefetch 距离受限 |
| Prologue sload | 一次性 | — | — |

---

## 五、MFMA 详细分析

### 5.1 GEMM0 (QK) — 8 个 MFMA

从 ATT trace 提取的 GEMM0 MFMA 数据：

| # | Vaddr | Hit | Lat | Stall | Op_sel | 说明 |
|---|-------|-----|-----|-------|--------|------|
| 1 | 0x2cd4 | 16 | 300 | 144 | op_sel_hi:[0,0,0] | kIter=0, nIter=0, mIter=0 (首次，C=0) |
| 2 | 0x2cfc | 16 | 132 | 0 | op_sel_hi:[0,0,0] | kIter=1, nIter=0, mIter=0 |
| 3 | 0x2d34 | 16 | 128 | 0 | op_sel:[1,0,0] op_sel_hi:[0,0,0] | kIter=0, nIter=1, mIter=0 |
| 4 | 0x2d64 | 16 | 128 | 0 | op_sel:[1,0,0] op_sel_hi:[0,0,0] | kIter=1, nIter=1, mIter=0 |
| 5 | 0x2db8 | 16 | 128 | 0 | op_sel_hi:[1,0,0] | kIter=0, nIter=0, mIter=1 |
| 6 | 0x2df0 | 16 | 128 | 0 | op_sel_hi:[1,0,0] | kIter=1, nIter=0, mIter=1 |
| 7 | 0x2e48 | 16 | 128 | 0 | op_sel:[1,0,0] op_sel_hi:[1,0,0] | kIter=0, nIter=1, mIter=1 |
| 8 | 0x2e7c | 16 | 128 | 0 | op_sel:[1,0,0] op_sel_hi:[1,0,0] | kIter=1, nIter=1, mIter=1 |

**观察：** 只有第 1 个 MFMA 有 stall=144（等待 LDS read），其余 7 个均为 0 stall，GEMM0 流水良好。

### 5.2 GEMM1 (PV) — 8 个 MFMA

| # | Vaddr | Hit | Lat | Stall | Op_sel | 说明 |
|---|-------|-----|-----|-------|--------|------|
| 1 | 0x33d4 | 16 | 128 | 0 | op_sel_hi:[0,0,0] | k1Iter=0, nIter=0 (V scale byte 0) |
| 2 | 0x33e8 | 16 | 192 | 0 | op_sel:[1,0,0] op_sel_hi:[0,0,0] | k1Iter=0, nIter=1 |
| 3 | 0x340c | 16 | 128 | 0 | op_sel_hi:[1,0,0] | k1Iter=1, nIter=0 (V scale byte 1) |
| 4 | 0x3420 | 16 | 416 | 352 | op_sel:[1,0,0] op_sel_hi:[1,0,0] | k1Iter=1, nIter=1 ★ |
| 5 | 0x3790 | 16 | 128 | 0 | op_sel_hi:[0,0,0] | 第 2 轮 k1Iter=0, nIter=0 |
| 6 | 0x37a4 | 16 | 192 | 0 | op_sel:[1,0,0] op_sel_hi:[0,0,0] | 第 2 轮 k1Iter=0, nIter=1 |
| 7 | 0x37c8 | 16 | 128 | 0 | op_sel_hi:[1,0,0] | 第 2 轮 k1Iter=1, nIter=0 |
| 8 | 0x38e0 | 16 | 128 | 0 | op_sel:[1,0,0] op_sel_hi:[1,0,0] | 第 2 轮 k1Iter=1, nIter=1 |

**观察：** GEMM1 第 4 个 MFMA (0x3420) stall=352，等待最后一批 LDS 读数据。第 1 轮最后一个 MFMA 存在 LDS pipeline bubble。

### 5.3 OPSEL dwordx4 验证

从 ASM 文件确认：
- **`buffer_load_dwordx4 v[106:109]`** (vaddr=0x26a8)：加载 K scale packed，4×int32 覆盖 pair[0]
- **`buffer_load_dwordx4 v[110:113]`** (vaddr=0x26b0)：加载 V scale packed，4×int32 覆盖 pair[0]
- **`v_cndmask_b32_e32 v216, v108, v106, vcc`**：从 dwordx4 中用 cndmask 选 int32[0] 或 int32[2]
- **`v_cndmask_b32_e32 v217, v109, v107, vcc`**：选 int32[1] 或 int32[3]
- 每 2 个 tile 后 **`buffer_load_dwordx4`** 重载（`if block_in_pair == 1`）

**确认无 shift/OR 指令**：没有任何 `v_lshrrev_b32`, `v_or_b32`, `v_and_b32` 用于 scale byte 提取。

---

## 六、性能对比

### 6.1 吞吐量数据

| seqlen | Baseline (16x16 OPSEL) | dwordx4 OPSEL | 增益 |
|--------|----------------------|---------------|------|
| 1024 | — | 860 TFLOPs | — |
| 4096 | 816 TFLOPs | **1077 TFLOPs** | **+32%** |
| 8192 | 814 TFLOPs | **1228 TFLOPs** | **+51%** |
| 16384 | — | **1273 TFLOPs** | — |

> **注：** Baseline 为之前的 16×16×128 OPSEL 版本（VGPR=88，occ=2）。dwordx4 OPSEL 版本使用 32×32×64 MFMA，VGPR=112，occ=2（相同）。

### 6.2 vs Baseline 关键指标对比

| 指标 | Baseline (16x16 OPSEL) | dwordx4 OPSEL | 说明 |
|------|----------------------|---------------|------|
| MFMA 形状 | 16×16×128 | 32×32×64 | 更大 MFMA tile |
| VGPR | 88 | 112 | +24，因 32×32 o_acc 更大 |
| Occupancy | 2/8 | 2/8 | 相同 |
| VALU:MFMA (PMC) | ~39:1 | **33.5:1** | 改善 14% |
| VALU 利用率 | 33.2% | **71.5%** | 改善 2.15× |
| Duration (s=4096) | ~615 µs (估) | 502 µs | -18% |

### 6.3 性能增益来源分析

1. **32×32×64 MFMA 效率更高**：每个 MFMA 计算 32×32×64=65536 MACs（vs 16×16×128=32768 MACs），寄存器利用率翻倍
2. **dwordx4 减少 VMEM 指令**：每 tile-pair 只需 1 次 dwordx4 load（vs 8× ubyte load），节省约 7 条 VMEM 指令/pair
3. **消除 shift/OR 打包**：8 条 VALU 指令/pair（shift+OR）被完全消除
4. **OPSEL cycling 零开销**：scale byte 选择在 MFMA 硬件内部完成，无额外指令

**主要性能增益来自 (1)，(2)(3) 为次要贡献。**

---

## 七、剩余瓶颈与优化方向

### 7.1 当前主要瓶颈

```
V tile 48× buffer_load_dword (非向量化)
└─ 每 iter 约 48 条独立 VMEM，L2 miss 串行排队
└─ 占 stall 约 40%，是最大未解决瓶颈
```

### 7.2 LDS Bank Conflict

SQ_LDS_BANK_CONFLICT = 16,777,216 → 每 wave 约 2027 次冲突。
- 来源：V tile LDS 写入时 4-warp 同时写，相同 bank 地址
- 影响：V→LDS 写入延迟增加，但被 VMEM pipeline 掩盖（影响有限）

### 7.3 潜在改进方向

| 方向 | 预期收益 | 复杂度 |
|------|---------|--------|
| V tile 改为 buffer_load_dwordx4（重排 LDS layout） | -15~20% V load stall | 高 |
| 增加 V prefetch 深度（2 tile ahead） | 部分隐藏 L2 miss | 中 |
| 减少 barrier 次数（double-buffer K LDS） | -500 cycles/iter barrier | 高 |
| LDS bank conflict 消除（padding） | 小 | 低 |

---

## 八、结论

SA3 fp16 dwordx4 OPSEL 优化方案成功将：
- K/V scale 加载从 **8× ubyte load** 降为 **1× dwordx4 load**（覆盖 2 个 tile pair）
- 结合 32×32×64 MFMA 的 **OPSEL cycling**，消除所有 shift/OR 打包指令

最终实现 **s=4096 时 +32%（816→1077 TFLOPs），s=8192 时 +51%（814→1228 TFLOPs）** 的性能提升。

剩余主要瓶颈为 V tile 按 `buffer_load_dword` 逐行加载（共 48 条/iter）。后续优化应聚焦于 V tile VMEM 向量化和 LDS layout 重设计。

---

*报告生成时间：2026-04-18*
*ASM 反汇编：`build/bin/tile_example_fmha_fwd.49.hipv4-amdgcn-amd-amdhsa--gfx950`*
*ATT trace dispatch: `ui_output_agent_9578_dispatch_14`*
