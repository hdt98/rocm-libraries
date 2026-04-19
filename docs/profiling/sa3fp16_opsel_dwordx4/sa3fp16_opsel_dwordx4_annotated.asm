; ==============================================================================
; SA3 fp16 FMHA Forward — dwordx4 OPSEL Scale 优化版本 带注释汇编
; ==============================================================================
; Kernel:  BlockFmhaPipelineQRKSVSSageAttn
;          TileFmhaShape<128,128,64,128,64,128>  (kM0=128, kN0=128, kK0=64, kN1=128, kK1=64)
;          MFMA: v_mfma_scale_f32_32x32x64_f8f6f4  (32x32x64, fp4 A/B, fp32 C)
;          WG=256 threads (4 warps), VGPR=112, LDS=4352B
; 优化:    K/V scale 通过 buffer_load_dwordx4 一次加载 4 个 int32（覆盖 2 个 tile-pair）
;          GEMM0/GEMM1 均使用 OPSEL cycling (op_sel:[0/1,0,0] / op_sel_hi:[0/1,0,0])
;          替代原来的 8×buffer_load_ubyte + shift/OR 打包。
;
; PMC (s=4096, b=2, h=32):
;   VGPR=112, LDS=4352B, WG=256 (4 warps), Duration=502us
;   SQ_WAVES:            8,280
;   SQ_INSTS_VALU:     144,884,346  (含 MFMA)
;   SQ_INSTS_MFMA:       4,194,304
;   SQ_INSTS_SALU:       4,073,017
;   SQ_INSTS_LDS:        5,767,168
;   SQ_INSTS_VMEM:      18,186,386
;   SQ_ACTIVE_INST_VALU:174,546,470
;   SQ_BUSY_CU_CYCLES:  244,037,525
;   SQ_LDS_BANK_CONFLICT: 16,777,216
;
; 计算派生指标:
;   纯 MFMA / wave:            507  (4,194,304 / 8,280)
;   纯 VALU(非MFMA) / wave:  17,000  ((144,884,346-4,194,304)/8,280)
;   VALU:MFMA 比               33.5:1  (较 baseline 改善: 之前约 39:1)
;   VALU 利用率:                59.7%  (SQ_ACTIVE_INST_VALU / SQ_BUSY_CU_CYCLES)
;   LDS bank conflict / wave: 2,027
;
; 性能对比 (d=128, b=2, h=32):
;   s=4096:  816 TFLOPs (baseline) → 1077 TFLOPs (dwordx4 OPSEL)  +32%
;   s=8192:  814 TFLOPs (baseline) → 1228 TFLOPs (dwordx4 OPSEL)  +51%
;   s=16384: -                     → 1273 TFLOPs
;
; SRD 映射 (s_load → buffer resource descriptor → buffer_load 使用):
;   s[52:55]  q_ptr (kargs+0x00)       → Q tile           2× dwordx4
;   s[24:27]  k_ptr (kargs+0x08)       → K tile           2× dwordx4
;   s[40:43]  v_ptr (kargs+0x10)       → V tile (fp4)     2× dwordx4
;   s[44:47]  delta_s_ptr (kargs+0xa8) → delta_s (fp32)   62× dword (stride-0 M-broadcast)
;   s[16:19]  k_scale_packed (kargs+0xc0) → K scale       2× dwordx4 (prefetch + reload)
;   s[48:51]  v_scale_packed (kargs+0xc8) → V scale + Q scale  2× dwordx4 + 2× ubyte
;
; 算法阶段划分:
;   §0  Prologue           — 加载 kernel args，计算 batch/head 偏移
;   §1  Q 加载             — Q tile 从 global memory 读入寄存器 (load once)
;   §2  外层 K 循环 per tile:
;       §2a  数据预取        — K tile (2× dwordx4 s[24:27])、delta_s (62× dword s[44:47])、
;                              V tile fp4 (2× dwordx4 s[40:43])，三者交叉发射隐藏延迟
;       §2b  GEMM0 (QK)     — 8× v_mfma_scale_f32_32x32x64_f8f6f4，OPSEL cycling
;       §2c  Softmax        — row-max reduce (XOR-sync), v_exp_f32, row-sum reduce
;       §2d  P 量化          — p_scale_factor 归一化，cast_tile_mx → fp4，P scale 计算
;       §2e  V→LDS+GEMM1    — V tile 从 VGPR 写入 LDS，GEMM1 从 LDS 读 V
;       §2f  GEMM1 (PV)     — 8× v_mfma_scale_f32_32x32x64_f8f6f4，OPSEL cycling
;       §2g  Scale 重载      — 每处理完 2 个 tile，重新 buffer_load_dwordx4 K/V scale
;   §3  O 归一化           — 乘以 inv_l，类型转换 fp32→fp16
;   §4  Epilogue           — O tile 写回 global memory
;
; 注: delta_s 使用 stride-0 M 维广播，load_tile 为 [kM0=128, kN0=128] 全 tile
;     生成 62 次 buffer_load_dword，同一 seqlen_k 位置被多线程重复加载 (L1 命中)
;     V tile 为 fp4 格式 (pk_float4_e2m1_t)，128×64 fp4 = 4096B，每线程 16B = 1× dwordx4
;     k1_loops=2 故 V tile 共 2× dwordx4，远少于 delta_s 的 62× dword
;
; ATT stall 注释格式: [hit=N stall=S lat=L] 其中 stall 单位为 GPU cycles
;   ★★★★  stall > 2000   严重瓶颈
;   ★★★   stall 1000-2000 较大瓶颈
;   ★★    stall  500-1000 中等
;   ★     stall  200-500  轻微
; ==============================================================================

./bin/tile_example_fmha_fwd.49.hipv4-amdgcn-amd-amdhsa--gfx950:	file format elf64-amdgpu

Disassembly of section .text:

; ==============================================================================
; §0  PROLOGUE — 加载 kernel args，计算 seqlen/head 偏移，初始化寄存器
; ==============================================================================
0000000000002100 <_ZN7ck_tile6kentryI...>:
	s_load_dwordx4 s[36:39], s[0:1], 0x28                      // 000000002100: C00A0900 00000028
	; [hit=4 lat=16 stall=0]    ; 加载 kernel arg: seqlen/head dims
	s_load_dwordx8 s[8:15], s[0:1], 0x3c                       // 000000002108: C00E0200 0000003C
	; [hit=4 lat=16 stall=0]    ; 加载 kernel arg: strides
	s_waitcnt lgkmcnt(0)                                        // 000000002110: BF8CC07F
	; [hit=4 lat=3940 stall=3940] ★★★★ 等待 sload 完成，prologue s_load latency
	s_add_i32 s5, s39, 0x7f                                     // 000000002114: 8105FF27 0000007F
	; [hit=4 stall=0]           ; ceil_div(seqlen_k, 128) 准备：+127
	s_ashr_i32 s6, s5, 31                                       // 00000000211C: 90069F05
	s_lshr_b32 s6, s6, 25                                       // 000000002120: 8F069906
	s_add_i32 s5, s5, s6                                        // 000000002124: 81050605
	s_ashr_i32 s28, s5, 7                                       // 000000002128: 901C8705
	; 计算 num_total_loop = ceil(seqlen_k / kN0=128)
	s_abs_i32 s5, s28                                           // 00000000212C: BE85301C
	v_cvt_f32_u32_e32 v1, s5                                    // 000000002130: 7E020C05
	s_mov_b32 s33, s39                                          // 000000002134: BEA10027
	s_load_dwordx2 s[6:7], s[0:1], 0x5c                        // 000000002138: C0060180 0000005C
	s_load_dwordx4 s[40:43], s[0:1], 0xd0                      // 000000002140: C00A0A00 000000D0
	v_rcp_iflag_f32_e32 v1, v1                                  // 000000002148: 7E024701
	s_load_dwordx8 s[16:23], s[0:1], 0xe4                      // 00000000214C: C00E0400 000000E4
	s_load_dwordx4 s[24:27], s[0:1], 0x100                     // 000000002154: C00A0600 00000100
	s_waitcnt lgkmcnt(0)                                        // 00000000215C: BF8CC07F
	; [hit=4 lat=5332 stall=5332] ★★★★ 等待多批次 sload，prologue 最大 stall
	; 计算 batch_idx * stride_b，head_idx * stride_h
	s_xor_b32 s23, s3, s28                                      // 000000002160: 88171C03
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                            // 000000002164: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                    // 00000000216C: 7E020F01
	s_ashr_i32 s23, s23, 31                                     // 000000002170: 90179F17
	; ... batch/head 索引整数除法 (使用 reciprocal 近似) ...

	; --- [许多整数除法/地址计算指令省略，共约 110 条 SALU] ---

	; 计算 Q/K/V/scale dram 指针偏移
	s_waitcnt lgkmcnt(0)                                        // 000000002414: BF8CC07F
	; [hit=4 lat=1740 stall=1740] ★★★ 等待最终 batch_idx lookup 地址 sload

	; ==============================================================================
	; §1  Q 加载 — 从 DRAM 读取 Q tile (kM0=128 × kK0=64) 到 VGPR
	;     同时从 DRAM 读取 Q scale tile
	;     dwordx4 版本：在此处预加载首批 K/V scale (buffer_load_dwordx4)
	; ==============================================================================
	; 计算 thread 分配：lane_id, warp_id → Q/K/V tile 地址
	v_mbcnt_lo_u32_b32 v1, -1, 0                                // 0000000024D8: D28C0001 000100C1
	v_mbcnt_hi_u32_b32 v121, -1, v1                             // 0000000024E0: D28D0079 000202C1
	; v121 = thread lane id in workgroup (0..255)

	; --- Q/Q-scale 地址计算约 30 条 VALU ---

	; Q tile 从 DRAM 加载 (s[52:55] = q_ptr SRD)
	buffer_load_dwordx4 v[98:101], v3, s[52:55], 0 offen        // 000000002698: E05C1000 800D6203
	; [hit=4 lat=56 stall=40]   ; Q tile 第 0 行 (128bit = 4×int32 fp4 data)
	buffer_load_dwordx4 v[102:105], v6, s[52:55], 0 offen       // 0000000026A0: E05C1000 800D6606
	; [hit=4 lat=20 stall=0]    ; Q tile 第 1 行

	; K scale dwordx4 预加载 (s[16:19] = k_scale_packed SRD)
	; 每个 int32 含 4 字节 OPSEL-packed e8m0 scale，dwordx4 覆盖 2 个 tile-pair
	buffer_load_dwordx4 v[106:109], v129, s[16:19], 0 offen     // 0000000026A8: E05C1000 80046A81
	; [hit=4 lat=20 stall=0]    ; K scale dwordx4: pair[0] 4×int32

	; V scale dwordx4 预加载 (s[48:51] = v_scale_packed SRD)
	buffer_load_dwordx4 v[110:113], v129, s[48:51], 0 offen     // 0000000026B0: E05C1000 800C6E81
	; [hit=4 lat=24 stall=8]    ; V scale dwordx4: pair[0] 4×int32

	; 初始化 o_acc[0..63]=0, m=-inf, l=0, cum_log_scale=0
	v_bfrev_b32_e32 v3, 0.5                                     // 0000000026BC: 7E0658F0
	s_mov_b32 s15, 0xff800000                                   // 0000000026CC: BE8F00FF FF800000
	; s15 = -infinity (fp32 bit pattern)
	v_mov_b32_e32 v17, 0                                        // 0000000026C8: 7E220280
	; v17 = 0.0f (用于批量初始化 o_acc)

	; o_acc 初始化 (v2..v17, v18..v33, v34..v49, v50..v65)
	v_mov_b32_e32 v16, v17                                      // 0000000026D8: 7E200311
	v_mov_b32_e32 v15, v17                                      // 0000000026DC: 7E1E0311
	; ... 共 64× v_mov 初始化 o_acc 所有累加器寄存器 ...
	; [全部 stall=0，流水友好]

	; m (row max) 初始化: v125 = -inf
	v_mov_b32_e32 v125, 0xff800000                              // 0000000027F8: 7EFA02FF FF800000

	; 外层循环开始前设置循环变量
	; s21 = i_total_loops = 0
	; s_branch 进入 Q-load loop (加载 Q scale)

	; ==============================================================================
	; §2  外层 K 循环（每次迭代处理 1 个 kN0=128 的 K/V tile）
	; ==============================================================================

	; ────────────────────────────────────────────────────────────────────
	; §2a  数据预取 — K tile (s[24:27]), delta_s (s[44:47]), V tile fp4 (s[40:43])
	;      三种 DRAM load 交叉发射，利用 VMEM FIFO 深度隐藏延迟
	; ────────────────────────────────────────────────────────────────────

	; 从 dwordx4 寄存器中提取当前 tile 的 K scale
	; block_in_pair = i_total_loops % 2
	; k_scale_ki0 = (block_in_pair==0) ? k_scale_4[0] : k_scale_4[2]
	; k_scale_ki1 = (block_in_pair==0) ? k_scale_4[1] : k_scale_4[3]
	s_bitcmp0_b32 s14, 0                                        // 00000000283C: BF0C800E
	s_cselect_b64 vcc, -1, 0                                    // 000000002840: 85EA80C1
	; vcc = (i_total_loops % 2 == 0) ? all-ones : 0

	; K tile 第 0 段 (s[24:27] = k_ptr SRD)
	buffer_load_dwordx4 v[66:69], v66, s[24:27], 0 offen        // 00000000286C: E05C1000 80064242
	; [hit=16 lat=1228 stall=1164] ★★★ K tile 加载，L2 miss (K 每 iter 地址不同)

	; delta_s 加载 (s[44:47] = delta_s_ptr SRD, float32)
	; delta_s shape=[1, kN0=128] 但使用 stride-0 M 维广播 → load_tile 生成 62× buffer_load_dword
	; 同一 seqlen_k 位置被多线程重复加载，依赖 L1 cache 命中
	buffer_load_dword v154, v71, s[44:47], 0 offen              // 000000002998: E0501000 800B9A47
	; [hit=16 lat=1040 stall=976] ★★★ delta_s 首批 dword，冷启动 L2 miss
	buffer_load_dword v155, v72, s[44:47], 0 offen              // 0000000029A0: E0501000 800B9B48
	buffer_load_dword v156, v73, s[44:47], 0 offen              // 0000000029A8: E0501000 800B9C49
	buffer_load_dword v157, v74, s[44:47], 0 offen              // 0000000029B0: E0501000 800B9D4A
	buffer_load_dword v158, v75, s[44:47], 0 offen              // 0000000029B8: E0501000 800B9E4B
	buffer_load_dword v159, v76, s[44:47], 0 offen              // 0000000029C0: E0501000 800B9F4C
	; [hit=16 lat=888 stall=824] ★★ delta_s L1 miss (不同线程首次访问)
	; ... 共 62× buffer_load_dword 加载 delta_s (stride-0 广播导致高指令数)
	buffer_load_dword v160, v77, s[44:47], 0 offen              // 0000000029C8: E0501000 800BA04D
	buffer_load_dword v161, v78, s[44:47], 0 offen              // 0000000029D0: E0501000 800BA14E
	buffer_load_dword v162, v79, s[44:47], 0 offen              // 0000000029D8: E0501000 800BA24F
	buffer_load_dword v163, v80, s[44:47], 0 offen              // 0000000029E0: E0501000 800BA350
	buffer_load_dword v164, v81, s[44:47], 0 offen              // 0000000029E8: E0501000 800BA451
	buffer_load_dword v165, v82, s[44:47], 0 offen              // 0000000029F0: E0501000 800BA552
	buffer_load_dword v166, v83, s[44:47], 0 offen              // 0000000029F8: E0501000 800BA653
	buffer_load_dword v167, v84, s[44:47], 0 offen              // 000000002A00: E0501000 800BA754
	; [hit=16 lat=1040 stall=976] ★★★ delta_s 加载 stall (VMEM FIFO 深度制约)
	buffer_load_dword v168, v85, s[44:47], 0 offen              // 000000002A08: E0501000 800BA855
	buffer_load_dword v169, v86, s[44:47], 0 offen              // 000000002A10: E0501000 800BA956
	; ... [省略中间 30+ 条 buffer_load_dword delta_s] ...
	buffer_load_dword v215, v141, s[44:47], 0 offen             // 000000002C78: E0501000 800BD78D

	; V tile fp4 预取 第 1 次 (s[40:43] = v_ptr SRD, pk_float4_e2m1_t)
	; V tile [kN1=128, kK1=64] fp4 = 4096B，每线程 16B = 1× dwordx4
	buffer_load_dwordx4 v[114:117], v77, s[40:43], 0 offen      // 000000002C80: E05C1000 800A724D
	; [hit=16 lat=296 stall=232] ★★ V tile fp4 dwordx4 (k1_loop=0)

	; K tile 第 1 段 (s[24:27])
	buffer_load_dwordx4 v[70:73], v70, s[24:27], 0 offen        // 000000002A68: E05C1000 80064646
	; [hit=16 lat=408 stall=344] ★★ K tile 第 2 个 dwordx4

	; (delta_s 62× dword loads 完成)

	s_waitcnt vmcnt(1)                                          // 00000000294C: BF8CC07F
	; [hit=16 lat=4684 stall=4684] ★★★★ 最大 per-iter stall: 等待 delta_s 62× dword loads
	;   原因: 62 条 buffer_load_dword 占满 VMEM FIFO，ds_write 前必须等前批就绪

	; K 第 1 段存 LDS
	ds_write_b128 v74, v[66:69]                                 // 000000002AD4: D9BE0000 0000424A
	; [hit=16 stall=0]           ; K[0..3] → LDS

	s_waitcnt vmcnt(33)                                         // 000000002AD0: BF8C8F71
	; [hit=16 lat=3748 stall=3748] ★★★★ 等待 delta_s 前 33 个 dword 就绪

	; K 第 2 段存 LDS
	ds_write_b128 v74, v[70:73]                                 // 000000002CBC: D9BE0000 0000464A
	; [hit=16 stall=0]           ; K[4..7] → LDS

	s_waitcnt lgkmcnt(0)                                        // 000000002C88: BF8CC07F
	s_barrier                                                   // 000000002C8C: BF8A0000
	; [hit=16 lat=1240 stall=1240] ★★★ 等待 LDS 写完成 + barrier 同步
	;   4 个 warp 同步点，barrier overhead 约 1240 cycles

	; ────────────────────────────────────────────────────────────────────
	; §2b  GEMM0 (QK) — 8× v_mfma_scale_f32_32x32x64_f8f6f4
	;      Q (fp4, AREG) × K (fp4, LDS/BREG) → s_acc (fp32)
	;      OPSEL cycling: op_sel_hi:[0/1,0,0] 选择 Q scale 的 byte 0/1
	;      K scale 从 dwordx4 寄存器直接取 int32 (无 shift/OR 开销)
	; ────────────────────────────────────────────────────────────────────

	; 从 LDS 读 K tile (4× ds_read_b128, 分两次同步)
	ds_read_b128 v[66:69], v131                                 // 000000002C90: D9FE0000 42000083
	ds_read_b128 v[82:85], v132                                 // 000000002C98: D9FE0000 52000084
	ds_read_b128 v[142:145], v133                               // 000000002CA0: D9FE0000 8E000085
	ds_read_b128 v[150:153], v134                               // 000000002CA8: D9FE0000 96000086

	s_waitcnt lgkmcnt(0)                                        // 000000002CB0: BF8CC07F
	; [hit=16 lat=1200 stall=1200] ★★★ 等待 4× ds_read 完成 (LDS → VGPR)
	s_barrier                                                   // 000000002CB4: BF8A0000

	; GEMM0 MFMA #1 (nIter=0, mIter=0, kIter=0): op_sel_hi:[0,0,0] Q scale byte-0
	s_waitcnt lgkmcnt(0)                                        // 000000002CC4: BF8CC07F
	; [hit=16 lat=1144 stall=1144] ★★★ 等待第 2 轮 ds_read (K 第 2 段)

	v_cndmask_b32_e32 v216, v108, v106, vcc                    // 000000002CCC: 01B0D56C
	; v216 = (block_in_pair==0) ? k_scale_4[0] : k_scale_4[2]  (k_scale_ki0)
	s_nop 1                                                     // 000000002CD0: BF800001
	v_mfma_scale_f32_32x32x64_f8f6f4 v[66:81], v[66:69], v[98:101], 0, v216, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=300 stall=144] ; GEMM0 MFMA #1 (首次，无前置 C 累加)
	; v[66:81]=C_out, v[66:69]=A(K from LDS), v[98:101]=B(Q row0), v216=K_scale, v126=Q_scale
	; op_sel_hi:[0,0,0] → Q scale 使用 byte 0 (q_scale_ki 的低字节)
	; cbsz:4=fp4 A, blgp:4=fp4 B; 32×32×64=65536 MACs per warp

	ds_read_b128 v[86:89], v131                                 // 000000002CE4: D9FE0000 56000083
	v_cndmask_b32_e32 v217, v109, v107, vcc                    // 000000002CEC: 01B2D76D
	; v217 = (block_in_pair==0) ? k_scale_4[1] : k_scale_4[3]  (k_scale_ki1)
	ds_read_b128 v[146:149], v132                               // 000000002CF0: D9FE0000 92000084
	s_waitcnt lgkmcnt(1)                                        // 000000002CF8: BF8CC17F

	v_mfma_scale_f32_32x32x64_f8f6f4 v[66:81], v[86:89], v[102:105], v[66:81], v217, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=132 stall=0]  ; GEMM0 MFMA #2 (kIter=1, 累加到 v[66:81])
	; v217=k_scale_ki1, v127=Q_scale (byte 0, k1 段)
	; s_nop 11 = MFMA 延迟填充，避免 WAR hazard
	s_nop 11                                                    // 000000002D0C: BF80000B

	; (v[218:221] = delta_s 加法: s_acc += delta_s_tile 元素)
	v_add_f32_e32 v218, v154, v66                              // 000000002D10: 03B4859A
	v_add_f32_e32 v219, v156, v68                              // 000000002D14: 03B6899C
	v_add_f32_e32 v158, v158, v70                              // 000000002D18: 033C8D9E
	; ... (16× v_add_f32 累加 delta_s correction 到 s_acc 行) ...

	v_mfma_scale_f32_32x32x64_f8f6f4 v[82:97], v[82:85], v[98:101], 0, v216, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #3 (nIter=1, mIter=0, kIter=0)
	; op_sel:[1,0,0] → B scale (Q scale) 使用 byte 1 (同一 int32 的第 2 字节)
	; OPSEL cycling: nIter 0,1,2,3 对应 op_sel 0,1,2,3 → 4 个 Q-scale tile 共享 1 个 int32

	; (16× v_add_f32 省略)
	s_waitcnt lgkmcnt(0)                                        // 000000002D60: BF8CC07F

	v_mfma_scale_f32_32x32x64_f8f6f4 v[82:97], v[146:149], v[102:105], v[82:97], v217, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #4 (nIter=1, kIter=1)
	s_nop 11                                                    // 000000002D74: BF80000B

	; (v_add_f32 for GEMM0 output rows 处理省略)

	v_mfma_scale_f32_32x32x64_f8f6f4 v[66:81], v[142:145], v[98:101], 0, v216, v126 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #5 (nIter=0, mIter=1, kIter=0)
	; op_sel_hi:[1,0,0] → A scale (K scale) 使用 byte 1

	v_mfma_scale_f32_32x32x64_f8f6f4 v[66:81], v[82:85], v[102:105], v[66:81], v217, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #6 (nIter=0, mIter=1, kIter=1)

	v_mfma_scale_f32_32x32x64_f8f6f4 v[82:97], v[150:153], v[98:101], 0, v216, v126 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #7 (nIter=1, mIter=1, kIter=0)

	s_waitcnt vmcnt(17)                                         // 000000002E70: BF8C4F71
	; 等待剩余 delta_s dword + V tile dwordx4 就绪
	v_mfma_scale_f32_32x32x64_f8f6f4 v[82:97], v[154:157], v[102:105], v[82:97], v217, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM0 MFMA #8 (nIter=1, mIter=1, kIter=1)
	; ── GEMM0 完成: 8 个 MFMA，计算 s_acc = Q × K^T (fp4×fp4→fp32)
	; ── 关键: 只用 2 个 int32 (k_scale_ki0, k_scale_ki1) 覆盖所有 8 个 MFMA
	; ── 无 shift/OR 指令，OPSEL 在硬件内部选 byte

	; ────────────────────────────────────────────────────────────────────
	; §2c  Softmax — scale_s × s_acc，row-max reduce (XOR cross-warp),
	;      deferred rescale exp2 (fast_exp2 path), row-sum reduce
	; ────────────────────────────────────────────────────────────────────

	; 在 GEMM0 nop 间隙 穿插进行的地址计算
	s_waitcnt lgkmcnt(0)                                        // 000000002F18: BF8CC07F

	; row-max reduce: v_max3_f32 链 (32 个 s_acc 元素 → 1 个 max)
	v_max3_f32 v67, v218, s15, v220                            // 000000002F1C: D1D30043 07701FDA
	; s15 = -inf (初始 max)
	v_max3_f32 v67, v67, v219, v221                            // 000000002F24: D1D30043 0777B743
	; ... (共 16× v_max3_f32) ...
	v_max3_f32 v67, v67, v75, v84                              // 000000002FFC: D1D30043 05529743
	; [hit=16 lat=2924 stall=2924 idle=0] — 注: ATT stall 为全序列 max3 chain 终点

	; XOR cross-warp reduce: ds_bpermute_b32
	ds_bpermute_b32 v87, v135, v67                             // 00000000301C: D87E0000 57004387
	; v135 = XOR lane address (lane ^ 32)，与对称 warp 交换 max 值
	s_barrier                                                   // 000000003024: BF8A0000

	; 读取 cross-warp 传来的 max，合并得全局行 max
	s_waitcnt lgkmcnt(1)                                        // 00000000303C: BF8CC17F
	v_max3_f32 v125, v66, v67, v87                             // 000000003040: D1D3007D 055E8742
	; v125 = max(m_old, m_local, bperm_m_local) = new row max

	; v[66:67] = scale_s * m_new (对应 cum_log_scale update)
	v_pk_mul_f32 v[66:67], s[8:9], v[124:125]                  // 00000000305C: D3B14042 1802F808
	; s[8:9] = scale_s (fp32 scalar broadcast)

	; p_compute = exp2(scale_s * s - scale_s * m - cum_log_scale)
	; 实现: v_fma_f32 v87 = s9 * s[i] - v187  (v187 = row_base)
	;       v_exp_f32_e32 v87           (exp 或 exp2，fast path)
	v_fma_f32 v87, s9, v218, -v187                             // 00000000306C: D1CB0057 86EFB409
	v_exp_f32_e32 v87, v87                                     // 000000003074: 7EAE4157
	v_fma_f32 v89, s9, v220, -v187                             // 000000003078: D1CB0059 86EFB809
	v_exp_f32_e32 v89, v89                                     // 000000003080: 7EB24159
	; ... (共 32× v_fma_f32 + 32× v_exp_f32 = 64 指令 softmax main loop) ...
	; 注: exp_f32 使用 Pipe0 (Trans ALU)，与 MFMA (Pipe1 XDL) 可 co-execute

	v_exp_f32_e32 v168, v168                                   // 0000000031E8: 7F5041A8

	; ────────────────────────────────────────────────────────────────────
	; §2d  P 量化 — p_norm = p_compute × p_scale_factor
	;      cast_tile_mx: fp32 → fp4 (MXFP4)，计算 P scale (e8m0)
	; ────────────────────────────────────────────────────────────────────

	; row-sum reduce: 32× v_add_f32 (p_compute 元素求和 → l)
	v_mul_f32_e32 v169, s37, v87                               // 0000000031EC: 0B52AE25
	; s37 = p_scale_factor (用于 p_norm 归一化)
	v_mul_f32_e32 v174, s37, v89                               // 0000000031F0: 0B5CB225
	; p_norm 元素: v169, v174, v175, v176, ...

	; 计算 p_scale: max(|p_norm|) → e8m0 encoding
	v_max3_f32 v170, |v169|, 0, |v174|                         // 0000000031FC: D1D305AA 06B901A9
	; ... (16× v_max3_f32 计算 P tile 行最大绝对值) ...
	v_mul_f32_e32 v189, 0x3e2aaaab, v189                       // 000000003304: 0B7B7AFF 3E2AAAAB
	; × (1/6) 转换为 e8m0 scale factor

	v_add_u32_e32 v189, 0x7fffff, v189                         // 00000000330C: 697B7AFF 007FFFFF
	v_and_b32_e32 v208, 0xff800000, v189                       // 000000003314: 27A17AFF FF800000
	; v208 = P scale (e8m0 格式，float32 bit-extracted)

	; V tile 从 VGPR 写入 LDS (在 P 量化时穿插执行，V 数据来自 §2a 的 dwordx4 预取)
	s_waitcnt lgkmcnt(0)                                        // 000000003380: BF8CC07F
	s_barrier                                                   // 000000003384: BF8A0000
	ds_read_b128 v[178:181], v136                               // 000000003388: D9FE0000 B2000088
	; V tile 从 LDS 读出 (已在 softmax 后存入)

	; fp4 量化: v_cvt_scalef32_pk_fp4_f32 (32× 指令)
	v_cvt_scalef32_pk_fp4_f32 v174, v169, v174, v208           // 00000000331C: D23D00AE 07435DA9
	; [v174] = fp4_pack(v169, v174, scale=v208)，每条量化 2 个 fp32 值
	v_cvt_scalef32_pk_fp4_f32 v174, v175, v176, v208 op_sel:[0,0,1,0]
	; op_sel 控制输出写入 byte 位置 (打包为 fp4 pair)
	; ... (共 16× v_cvt_scalef32_pk_fp4_f32 完成 P tile 量化) ...
	v_cvt_scalef32_pk_fp4_f32 v177, v206, v207, v208 op_sel:[0,0,1,1]
	; P scale (e8m0): v_bfe_u32 提取 float 指数
	v_bfe_u32 v169, v189, 23, 8                                // 0000000033BC: D1C800A9 02212FBD
	; v169 = P scale byte (e8m0 exponent for first group)

	; ────────────────────────────────────────────────────────────────────
	; §2f  GEMM1 (PV) — 8× v_mfma_scale_f32_32x32x64_f8f6f4
	;      P (fp4, AREG) × V (fp4, 从 LDS 读取) → o_acc (fp32)
	;      V scale 从 dwordx4 中选 2 个 int32 (block_in_pair 决定 [0]/[2])
	;      P scale 从 v_bfe_u32 提取的 e8m0 byte
	;      OPSEL cycling: op_sel/op_sel_hi 选择 scale byte 0/1/2/3
	; ────────────────────────────────────────────────────────────────────

	; 读 V tile from LDS
	v_cndmask_b32_e32 v188, v112, v110, vcc                    // 0000000033C4: 0178DD70
	; v188 = (block_in_pair==0) ? v_scale_4[0] : v_scale_4[2]  (v_scale_arr[0])

	s_waitcnt lgkmcnt(1)                                        // 0000000033D0: BF8CC17F
	v_mfma_scale_f32_32x32x64_f8f6f4 v[50:65], v[178:181], v[174:177], v[50:65], v188, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM1 MFMA #1 (nIter=0, kIter=0)
	; v188 = v_scale_arr[0]，v169 = P scale (byte 0)
	; op_sel_hi:[0,0,0] → A scale (P scale) byte 0

	s_waitcnt lgkmcnt(0)                                        // 0000000033E4: BF8CC07F
	v_mfma_scale_f32_32x32x64_f8f6f4 v[34:49], v[182:185], v[174:177], v[34:49], v188, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=192 stall=0]  ; GEMM1 MFMA #2 (nIter=1, kIter=0)
	; op_sel:[1,0,0] → B scale (V scale) 使用 byte 1 of v188

	; 读 V tile (第 2 段) from LDS
	ds_read_b128 v[178:181], v138                               // 0000000033F8: D9FE0000 B200008A
	ds_read_b128 v[182:185], v139                               // 000000003400: D9FE0000 B600008B
	s_waitcnt lgkmcnt(1)                                        // 000000003408: BF8CC17F
	; [hit=16 lat=928 stall=928] ★★ GEMM1 LDS → VGPR 等待 (V 第 3 段)

	v_mfma_scale_f32_32x32x64_f8f6f4 v[18:33], v[178:181], v[174:177], v[18:33], v188, v169 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM1 MFMA #3 (nIter=0, kIter=1)
	; op_sel_hi:[1,0,0] → A scale (P scale) byte 1

	s_waitcnt lgkmcnt(0)                                        // 00000000341C: BF8CC07F
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[182:185], v[174:177], v[2:17], v188, v169 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=416 stall=352] ★★ GEMM1 MFMA #4 — 末尾等待 LDS
	; (第 kIter=1, nIter=1)

	; 第 2 个 V scale (v_scale_arr[1] 对应 kIter=1)
	; 通过 v_cndmask 从 dwordx4 选 byte 2/3
	v_bfe_u32 v181, v203, 23, 8                                // 00000000377C: D1C800B5 02212FCB
	; v181 = second P scale byte (e8m0, group 1)

	; GEMM1 MFMA #5-8 (对应 k1_loops 第 2 轮)
	s_waitcnt lgkmcnt(1)                                        // 00000000378C: BF8CC17F
	; [hit=16 lat=928 stall=928] ★★ 等待 LDS (第 4 段 V)

	v_mfma_scale_f32_32x32x64_f8f6f4 v[50:65], v[74:77], v[70:73], v[50:65], v188, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM1 MFMA #5 (k1-iter1, nIter=0)

	s_waitcnt lgkmcnt(0)                                        // 0000000037A0: BF8CC07F
	v_mfma_scale_f32_32x32x64_f8f6f4 v[34:49], v[78:81], v[70:73], v[34:49], v188, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; [hit=16 lat=192 stall=0]  ; GEMM1 MFMA #6

	ds_read_b128 v[74:77], v138                                 // 0000000037B4: D9FE0000 4A00008A
	ds_read_b128 v[78:81], v139                                 // 0000000037BC: D9FE0000 4E00008B
	s_waitcnt lgkmcnt(1)                                        // 0000000037C4: BF8CC17F
	; [hit=16 lat=928 stall=928] ★★ 等待 LDS (最后一段 V)

	v_mfma_scale_f32_32x32x64_f8f6f4 v[18:33], v[74:77], v[70:73], v[18:33], v188, v181 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM1 MFMA #7

	; row-sum reduce (l update): 47× v_add_f32 (p_compute 求和)
	v_add_f32_e32 v74, v87, v89                                // 0000000037D8: 0294B357
	v_add_f32_e32 v74, v91, v74                                // 0000000037DC: 0294955B
	; ... (共 47× v_add_f32) ...
	v_add_f32_e32 v68, v180, v68                               // 0000000038D0: 028889B4

	; XOR cross-warp reduce for l
	ds_bpermute_b32 v69, v135, v68                             // 0000000038D4: D87E0000 45004487
	; 对称 warp 交换 row-sum

	s_waitcnt lgkmcnt(1)                                        // 0000000038DC: BF8CC17F
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[78:81], v[70:73], v[2:17], v188, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; [hit=16 lat=128 stall=0]  ; GEMM1 MFMA #8 (最后一个)
	; ── GEMM1 完成: 8 个 MFMA，累加 PV → o_acc

	; ────────────────────────────────────────────────────────────────────
	; §2g  Scale 重载 — 每处理完 2 个 tile (block_in_pair==1)
	;      buffer_load_dwordx4 重新加载下一 pair 的 K/V scale
	;      K scale: s[16:19], V scale: s[48:51]
	; ────────────────────────────────────────────────────────────────────
	s_waitcnt lgkmcnt(0)                                        // 0000000038F0: BF8CC07F
	s_barrier                                                   // 0000000038F4: BF8A0000
	; 循环尾部: i_total_loops++，检查是否需要重载 scale dwordx4
	; if (block_in_pair == 1 && i+1 < num_total_loop):
	;   k_scale_4 = buffer_load_dwordx4(scale_pair_idx++)
	;   v_scale_4 = buffer_load_dwordx4(scale_pair_idx)
	; (下一 iter 开始时使用新的 dwordx4 寄存器值)

	; 循环控制: s_cbranch 回到 §2a
	s_and_b64 vcc, exec, vcc                                   // 0000000038F8: 86EA6A7E
	s_cbranch_vccnz 64449                                       // 0000000038FC: BF87FBC1
	s_add_i32 s21, s21, 1                                       // 000000003900: 81158115
	s_cmp_ge_i32 s14, s6                                        // 000000003904: BF03060E
	s_cbranch_scc1 64446                                        // 000000003908: BF85FBBE
	; 跳回外层循环头 (0x2804)

	; ==============================================================================
	; §3  O 归一化 — o_acc *= inv_l，类型转换 fp32 → fp16
	; ==============================================================================
	; 计算 inv_l = 1.0 / l (高精度除法: div_scale + rcp + fma + div_fixup)
	v_div_scale_f32 v66, s[0:1], s37, s37, 1.0                 // 000000003928: D1E00042 03C84A25
	v_rcp_f32_e32 v67, v66                                      // 000000003930: 7E864542
	v_div_scale_f32 v68, vcc, 1.0, s37, 1.0                    // 000000003934: D1E06A44 03C84AF2
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3             // 00000000393C: BA000901 00000003
	v_fma_f32 v69, -v66, v67, 1.0                              // 000000003944: D1CB0045 23CA8742
	v_fmac_f32_e32 v67, v69, v67                               // 00000000394C: 76868745
	v_mul_f32_e32 v69, v68, v67                                 // 000000003950: 0A8A8744
	v_fma_f32 v70, -v66, v69, v68                              // 000000003954: D1CB0046 25128B42
	v_fmac_f32_e32 v69, v70, v67                               // 00000000395C: 768A8746
	v_fma_f32 v66, -v66, v69, v68                              // 000000003960: D1CB0042 25128B42
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0             // 000000003968: BA000901 00000000
	v_div_fmas_f32 v66, v66, v67, v69                          // 000000003970: D1E20042 05168742
	v_div_fixup_f32 v66, v66, s37, 1.0                         // 000000003978: D1DE0042 03C84B42
	; 类似流程再计算 inv_p_scale_factor / l

	; o_acc 归一化: fp32 × inv_l，然后 fp32 → fp16 packed
	; v_fma_mixlo_f16: 混精度 FMA，fp32 → fp16 in-place
	v_fma_mixlo_f16 v67, v66, v50, 0                           // 000000003A20: D3A10043 02026542
	v_mov_b32_e32 v50, v51                                      // 000000003A28: 7E640333
	v_pk_mul_f32 v[50:51], v[66:67], v[50:51] op_sel_hi:[0,1]  // 000000003A30: D3B14032 10026542
	v_cvt_pk_f16_f32 v52, v50, v51                             // 000000003A38: D2670034 00026732
	; ... (共 32× v_fma_mixlo_f16 + pack 指令，O tile fp32→fp16 packed) ...

	; pack 成 2fp16 per dword (v_pack_b32_f16, v_alignbit_b32)
	v_pack_b32_f16 v2, v67, v52                                // 000000003CA0: D2A00002 00026943
	v_alignbit_b32 v3, v53, v52, 16                            // 000000003CA8: D1CE0003 02426935
	; ... (共 32× pack/align) ...

	; ==============================================================================
	; §4  EPILOGUE — O tile 写回 global memory (8× buffer_store_dwordx4)
	; ==============================================================================
	; 计算 O 的 global memory 地址 (batch/head/seq 偏移)
	v_readfirstlane_b32 s2, v0                                  // 000000003DA0: 7E040500
	v_lshlrev_b32_e32 v0, 1, v121                              // 000000003DA4: 2400F281
	; ... 地址计算 ...
	v_add_lshl_u32 v0, v0, v1, 1                               // 000000003DE0: D1FE0000 02060300
	; O tile 写回 (每次 dwordx4 = 4× fp16 pairs = 8 个 fp16)
	buffer_store_dwordx4 v[2:5], v0, s[0:3], 0 offen           // 000000003DE8: E07C1000 80000200
	; [hit=4 lat=3884 stall=3864] ★★★★ Epilogue store stall (L2 write latency)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen offset:16  // 000000003DF0: E07C1010 80000600
	buffer_store_dwordx4 v[10:13], v0, s[0:3], 0 offen offset:32// 000000003DF8: E07C1020 80000A00
	buffer_store_dwordx4 v[14:17], v0, s[0:3], 0 offen offset:48// 000000003E00: E07C1030 80000E00
	buffer_store_dwordx4 v[18:21], v0, s[0:3], 0 offen offset:64// 000000003E08: E07C1040 80001200
	buffer_store_dwordx4 v[22:25], v0, s[0:3], 0 offen offset:80// 000000003E10: E07C1050 80001600
	buffer_store_dwordx4 v[26:29], v0, s[0:3], 0 offen offset:96// 000000003E18: E07C1060 80001A00
	buffer_store_dwordx4 v[30:33], v0, s[0:3], 0 offen offset:112// 000000003E20: E07C1070 80001E00
	s_endpgm                                                    // 000000003E28: BF810000

; ==============================================================================
; TOP 15 STALL HOTSPOTS (ATT Trace Data)
; ==============================================================================
; Rank  Vaddr       Hit   Lat    Stall  Idle  Instruction / 原因
; ──────────────────────────────────────────────────────────────────────────────
;  #1   0x0000215c    4  5332   5332     0   s_waitcnt lgkmcnt(0)
;       原因: Prologue 多批次 s_load，等待 SGPR 全部从 L2/DRAM 返回
;
;  #2   0x0000294c   16  4684   4684     0   s_waitcnt vmcnt(1)
;       原因: delta_s 62× buffer_load_dword 占满 VMEM FIFO，等待前批就绪
;
;  #3   0x00002110    4  3940   3940     0   s_waitcnt lgkmcnt(0)
;       原因: Prologue 第 2 批 sload，batch_idx lookup 导致 DRAM read
;
;  #4   0x00003de8    4  3884   3864    32   buffer_store_dwordx4 v[2:5] (O write)
;       原因: Epilogue O tile store，L2 write latency + HBM 反压
;
;  #5   0x00002ad0   16  3748   3748     0   s_waitcnt vmcnt(33)
;       原因: 等待 delta_s 前 33 个 dword 就绪再写 K→LDS，VMEM FIFO 深度制约
;
;  #6   0x000022a8    4  1740   1740     0   s_waitcnt lgkmcnt(0)
;       原因: Prologue group_mode sload lookup 等待 (出现一次)
;
;  #7   0x00002c8c   16  1240   1240     0   s_barrier
;       原因: 4-warp barrier 同步点 (K 写 LDS 完成后)
;
;  #8   0x00002cb0   16  1200   1200     0   s_waitcnt lgkmcnt(0) [K LDS 读]
;       原因: 等待 4× ds_read_b128 完成，LDS → VGPR latency 约 1200 cycles/iter
;
;  #9   0x0000286c   16  1228   1164   128   buffer_load_dwordx4 v[66:69] (K tile)
;       原因: K tile 首次加载，L2 cache miss (每 iter K 地址不同)
;
; #10   0x00002cc4   16  1144   1144     0   s_waitcnt lgkmcnt(0) [K LDS 读第2批]
;       原因: 等待第 2 批 ds_read_b128 (K tile 第 2 段)
;
; #11   0x00002a00   16  1040    976     0   buffer_load_dword v167 (delta_s)
;       原因: delta_s dword 加载 L2 miss，地址每 iter 递增 kN0×sizeof(float)
;
; #12   0x00003408   16   928    928     0   s_waitcnt lgkmcnt(1) [GEMM1 LDS]
;       原因: 等待 ds_read_b128 (GEMM1 V tile 第 3 段)，LDS 延迟
;
; #13   0x000037c4   16   928    928     0   s_waitcnt lgkmcnt(1) [GEMM1 LDS]
;       原因: 等待 GEMM1 第 2 轮 ds_read_b128 (V 第 5 段)
;
; #14   0x000029c0   16   888    824     0   buffer_load_dword v159 (delta_s)
;       原因: delta_s 中段 dword load L2 miss
;
; #15   0x00002a40   16   872    808     0   buffer_load_dword v175 (delta_s)
;       原因: delta_s 末段 dword load L2 miss
; ==============================================================================
