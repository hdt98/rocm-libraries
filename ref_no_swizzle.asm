// ============================================================
// SubtileBasedKernel Assembly - FP4 64x64x256
// MFMA: v_mfma_f32_16x16x128_f8f6f4 (FP4)
// MIWaveGroup: [2, 2]
// BPE: 0.5   DepthU: 256
// ** NO ROTATION / NO SWIZZLING **
// ============================================================

// --- GRA Tile Assignment ---
/* GR Offset Calculation for Subtile Based Tiling */
v_lshrrev_b32 v65, 0x6, v[vgprSerial]              // Wave Id
v_and_b32 v66, v[vgprSerial], 63
v_and_b32 v60, v[vgprSerial], 7                    // get col_id in wave for 16B load
v_lshrrev_b32 v62, 0x3, v66                        // row id within wave
/* Swizzling/Rotation DISABLED */
v_mov_b32 v61, v60                                 // colIdB = colIdA (no swizzle)
s_mov_b32 s29, 32                                  // A: row offset
v_and_b32 v67, 0x1, v65                            // A: waveId % 2
v_lshrrev_b32 v68, 0x1, v65                        // A: waveId / 2
v_lshlrev_b32 v67, 0x3, v67                        // A: local row offset
v_mul_lo_u32 v68, s29, v68                         // A: wave row offset
v_add_u32 v63, v67, v68                            // A: row offset
s_mov_b32 s29, 32                                  // B: row offset
v_and_b32 v67, 0x1, v65                            // B: waveId % 2
v_lshrrev_b32 v68, 0x1, v65                        // B: waveId / 2
v_lshlrev_b32 v67, 0x3, v67                        // B: local row offset
v_mul_lo_u32 v68, s29, v68                         // B: wave row offset
v_add_u32 v64, v67, v68                            // B: row offset
v_add_u32 v63, v62, v63                            // A: row offset
v_lshlrev_b32 v68, 0x4, v60                        // scale col_id by load_width
v_mul_lo_u32 v67, s[sgprStrideA0I], v63            // A: rowId * stride
v_lshlrev_b32 v67, 0x2, v67                        // A: rowId*stride*bpe
v_lshrrev_b32 v67, 0x3, v67                        // to bytes
v_add_u32 v1, v68, v67                             // A: GR row_offset
v_add_u32 v64, v62, v64                            // B: row offset
v_lshlrev_b32 v68, 0x4, v61                        // scale col_id by load_width
v_mul_lo_u32 v67, s[sgprStrideB1J], v64            // B: rowId * stride
v_lshlrev_b32 v67, 0x2, v67                        // B: rowId*stride*bpe
v_lshrrev_b32 v67, 0x3, v67                        // to bytes
v_add_u32 v6, v68, v67                             // B: GR row_offset
s_mul_i32 s12, 0x8, s[sgprStrideA0I]               // A: 16 rows offset, stride 8, 1
s_mul_i32 s13, 0x8, s[sgprStrideB1J]               // B: 16 rows offset, stride 8, 1


// --- LRA Tile Assignment ---
/* LR Offset Calculation for Subtile Based Tiling */
v_and_b32 v61, v[vgprSerial], 63                   // laneId
v_lshrrev_b32 v61, 0x4, v61                        // lane16Group
v_and_b32 v60, v[vgprSerial], 15                   // laneId % 16
v_mov_b32 v64, v61                                 // colOffset = lane16Group
v_lshlrev_b32 v63, 0x7, v60                        // offsetRow = depthUBytes*lane16
v_mov_b32 v2, v64                                  // A: laneId
v_add_u32 v4, v2, 0x4                              // A: colOffset for MFMA 1 of subtile
v_and_b32 v4, v4, 0x7                              // A: colOffset = colOffset % block_size
v_lshlrev_b32 v2, 0x4, v2                          // A: colOffset*loadWidth
v_add_u32 v2, v2, v63                              // A: row + col
v_lshlrev_b32 v4, 0x4, v4                          // A: colOffset*loadWidth
v_add_u32 v4, v4, v63                              // A: row + col
v_mov_b32 v7, v64                                  // B: laneId
v_add_u32 v9, v7, 0x4                              // B: colOffset for MFMA 1 of subtile
v_and_b32 v9, v9, 0x7                              // B: colOffset = colOffset % block_size
v_lshlrev_b32 v7, 0x4, v7                          // B: colOffset*loadWidth
v_add_u32 v7, v7, v63                              // B: row + col
v_lshlrev_b32 v9, 0x4, v9                          // B: colOffset*loadWidth
v_add_u32 v9, v9, v63                              // B: row + col
v_lshrrev_b32 v11, 0x6, v[vgprSerial]              // waveId
v_and_b32 v11, 0x1, v11                            // A: waveId % 2
s_mov_b32 s29, 0x800                               // A: interleave stride
v_mul_lo_u32 v11, s29, v11
v_add_u32 v2, v2, v11                              // A: wave partition LR offset
v_add_u32 v4, v4, v11                              // A: wave partition LR offset
v_lshrrev_b32 v11, 0x6, v[vgprSerial]              // waveId
v_lshrrev_b32 v11, 0x1, v11                        // B: waveId / 2
s_mov_b32 s29, 0x800                               // B: interleave stride
v_mul_lo_u32 v11, s29, v11
v_add_u32 v7, v7, v11                              // B: wave partition LR offset
v_add_u32 v9, v9, v11                              // B: wave partition LR offset
v_add_u32 v7, 8192, v7                             // B matrix offset in LDS
v_add_u32 v9, 8192, v9                             // B matrix offset in LDS


// --- DTL Init ---
/* Compute shared offsets used by m0 in DTL loads */
v_lshrrev_b32 v11, 0x6, v[vgprSerial]              // Wave Id
s_mov_b32 s29, 32                                  // A: row offset
v_and_b32 v62, 0x1, v11                            // A: waveId % 2
v_lshrrev_b32 v63, 0x1, v11                        // A: waveId / 2
v_lshlrev_b32 v62, 0x3, v62                        // A: local row offset
v_mul_lo_u32 v63, s29, v63                         // A: wave row offset
v_add_u32 v60, v62, v63                            // A: row offset
s_mov_b32 s29, 32                                  // B: row offset
v_and_b32 v62, 0x1, v11                            // B: waveId % 2
v_lshrrev_b32 v63, 0x1, v11                        // B: waveId / 2
v_lshlrev_b32 v62, 0x3, v62                        // B: local row offset
v_mul_lo_u32 v63, s29, v63                         // B: wave row offset
v_add_u32 v61, v62, v63                            // B: row offset
v_lshlrev_b32 v60, 0x7, v60                        // Apply wave-specific offset for A
v_lshlrev_b32 v61, 0x7, v61                        // Apply wave-specific offset for B
s_nop 0                                            // Wait for VGPR to be ready
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrA], v60 // Store base LDS offset, will be modified
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrB], v61 // Store base LDS offset, will be modified
s_add_u32 s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteBaseAddrB], 0x2000
s_add_u32 s[sgprLocalWriteSwapA], s[sgprLocalWriteBaseAddrA], 16384
s_xor_b32 s[sgprLocalWriteSwapA], s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteSwapA]
s_add_u32 s[sgprLocalWriteSwapB], s[sgprLocalWriteBaseAddrB], 16384
s_xor_b32 s[sgprLocalWriteSwapB], s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteSwapB]


// --- Init D Accumulators to Zero ---
/* Init C vgprTiles to zero */
v_mov_b64 v[60:61], 0
s_nop 1                                            // wait for vgpr to be ready before MFMA
v_mfma_i32_32x32x16_i8 v[44:59], v[60:61], v[60:61], 0 // initC: [44:59]


// --- Main Loop ---
/* MAINLOOP */
/* REMOVE WHEN IMPLEMNTED: Placeholder for subtile based main loop impl */
label_start:
/* Emit load for A subtile: [0, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrA], 0
buffer_load_dwordx4 v1, s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // grBaseId = 0, i= 0
/* Emit load for A subtile: [1, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrA], 2048
buffer_load_dwordx4 v1, s[sgprSrdA:sgprSrdA+3], s12 offen offset:0 lds // grBaseId = 1, i= 0
/* Emit load for B subtile: [0, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrB], 0
buffer_load_dwordx4 v6, s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // grBaseId = 0, i= 0
/* Emit load for B subtile: [1, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrB], 2048
buffer_load_dwordx4 v6, s[sgprSrdB:sgprSrdB+3], s13 offen offset:0 lds // grBaseId = 1, i= 0
s_waitcnt vmcnt(0)                                 // Wait for all subtile GRs to complete
s_barrier
ds_read_b128 v[12:15], v2 offset:0                 // SubtileA[0,0] mfmaId=[0,0]
ds_read_b128 v[16:19], v4 offset:0                 // SubtileA[0,0] mfmaId=[0,1]
ds_read_b128 v[20:23], v2 offset:4096              // SubtileA[1,0] mfmaId=[0,0]
ds_read_b128 v[24:27], v4 offset:4096              // SubtileA[1,0] mfmaId=[0,1]
ds_read_b128 v[28:31], v7 offset:0                 // SubtileB[0,0] mfmaId=[0,0]
ds_read_b128 v[32:35], v9 offset:0                 // SubtileB[0,0] mfmaId=[0,1]
ds_read_b128 v[36:39], v7 offset:4096              // SubtileB[1,0] mfmaId=[0,0]
ds_read_b128 v[40:43], v9 offset:4096              // SubtileB[1,0] mfmaId=[0,1]
s_waitcnt lgkmcnt(0)                               // Wait for all subtile LRs to complete
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[12:15], v[28:31], v[44:47], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 0] += A[0, 0] * B[0, 0]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[20:23], v[28:31], v[48:51], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 0] += A[1, 0] * B[0, 0]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[12:15], v[36:39], v[52:55], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 1] += A[0, 0] * B[0, 1]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[20:23], v[36:39], v[56:59], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 1] += A[1, 0] * B[0, 1]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[16:19], v[32:35], v[44:47], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 0] += A[0, 1] * B[1, 0]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[24:27], v[32:35], v[48:51], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 0] += A[1, 1] * B[1, 0]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[16:19], v[40:43], v[52:55], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 1] += A[0, 1] * B[1, 1]
v_mov_b32 v11, 0x80                                // hardcoded scale 0x80
v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[24:27], v[40:43], v[56:59], v11, v11 cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 1] += A[1, 1] * B[1, 1]
/* Emit code to swap A GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteSwapA]
/* Emit code to swap B GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteSwapB]
/* Emit code to swap A LR vgpr offsets */
v_xor_b32 v2, v2, v3
v_xor_b32 v4, v4, v5
/* Emit code to swap B LR vgpr offsets */
v_xor_b32 v7, v7, v8
v_xor_b32 v9, v9, v10
s_add_u32 s[sgprSrdA], s[sgprSrdA], 128
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0
s_add_u32 s[sgprSrdB], s[sgprSrdB], 128
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc0 label_start
// 

