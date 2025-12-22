
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950
.globl Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950
.p2align 8
.type Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 352 // vgprs
  .amdhsa_next_free_sgpr 88 // sgprs
  .amdhsa_group_segment_fixed_size 88576 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 13
  .amdhsa_user_sgpr_kernarg_preload_length 11
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =256 */
/* Num AccVGPR=96 */
/* Num SGPR   =88 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 32 x 3 */
/* SubGroup= 4 x 64 */
/* VectorWidthA=8 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=0 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950
    .symbol: 'Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Gemm info
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info0
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info1
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            numWG
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            AddressWS
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressFlags
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      f32
      - .name:            ItersPerTile
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberItersPerTile
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicShiftItersPerTile
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            TotalIters
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SKItersPerWG
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skGrid
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skTiles
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          148
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          156
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          164
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          168
        .value_kind:      by_value
        .value_type:      u32
      - .name:            activationAlpha
        .size:            4
        .offset:          172
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationBeta
        .size:            4
        .offset:          176
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationType
        .size:            4
        .offset:          180
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   88576
    .kernarg_segment_align:      8
    .kernarg_segment_size:       184
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 88
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_unaligned_k_gfx950:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 24
.set vgprLocalWriteAddrA, 20
.set vgprLocalWriteAddrB, 21
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 8
.set vgprLocalReadAddrA, 22
.set vgprLocalReadAddrB, 23
.set vgprSerial, 192

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+64
.set vgprG2LA_BASE, vgprBase+88
.set vgprG2LB_BASE, vgprBase+120
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+12
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprStaggerU, 6
.set sgprWGM, 7
.set sgprLoopCounterL, 8
.set sgprOrigLoopCounter, 9
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 10
.set sgprNumWorkGroups1, 11
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprAddressWS, 32
.set sgprAddressFlags, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprItersPerTile, 46
.set sgprMagicNumberItersPerTile, 47
.set sgprMagicShiftItersPerTile, 48
.set sgprTotalIters, 49
.set sgprSKItersPerWG, 50
.set sgprskGrid, 51
.set sgprskTiles, 52
.set sgprStreamKIdx, 53
.set sgprStreamKIter, 54
.set sgprStreamKIterEnd, 55
.set sgprStreamKLocalStart, 56
.set sgprStreamKLocalEnd, 57
.set sgprSrdWS, 60

/* StreamK Parallel Reduction Assignments */
.set sgprSkSplit, sgprskTiles+0
.set sgprSkPartialIdx, sgprBeta+0

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 192
.set DepthU, 128
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
.set BpeAGR, 2
.set BpeAGRLog2, 1
.set BpeBGR, 2
.set BpeBGRLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x20000                           */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x20000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s16, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s19, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s17, s16, 0x1e                          // Get arg type
s_and_b32 s16, 0x3fffffff, s16                     // Get nums of gemm
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx16 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_waitcnt lgkmcnt(0)                               // preload
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 37 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s16, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s17, s2, 0x1e                           // Get arg type
s_mov_b32 s18, s3                                  // Preload internal args
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

.set DBG, 0

/* Load Kernel Args */
s_load_dword s27, s[sgprKernArgAddress:sgprKernArgAddress+1], 28 // 28
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 32 // 32
s_load_dwordx8 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_mov_b64 s[20:21], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[22:23], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[24:25], s[10:11]                       // move preload data to correct sgpr
s_mov_b32 s26, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s19, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s18, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_mov_b32 s[sgprArgType], s17
s_mov_b32 m0, 0x15a00                              // LDS clamp at 88576 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
.if DBG
s_cmp_eq_u32 s13, 1
s_cbranch_scc0 label_KernelEnd
s_branch label_skip_WGMXCC
.endif

/* remap workgroup to XCCs */
s_lshr_b32 s58, s[sgprWGM], 0x10                   // Get WGMXCC
s_and_b32 s58, s58, 0x3f                           // Get WGMXCC
/* remap WGs if WGMXCC > 1 */
s_cmp_gt_u32 s58, 1
s_cbranch_scc0 label_skip_WGMXCC
/* divmod(old_wg, WGMXCC) */
v_cvt_f64_u32 v[24:25], s58                        // s59 = s[sgprWorkGroup0] / s58
v_rcp_f64 v[24:25], v[24:25]                       // s59 = s[sgprWorkGroup0] / s58
v_cvt_f64_u32 v[26:27], s[sgprWorkGroup0]          // s59 = s[sgprWorkGroup0] / s58
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s59 = s[sgprWorkGroup0] / s58
v_cvt_u32_f64 v24, v[24:25]                        // s59 = s[sgprWorkGroup0] / s58
v_mul_lo_u32 v25, v24, s58                         // s59 = s[sgprWorkGroup0] / s58
v_sub_u32 v26, s[sgprWorkGroup0], v25              // s59 = s[sgprWorkGroup0] / s58
v_cmpx_ge_u32 exec, v26, s58                       // s59 = s[sgprWorkGroup0] / s58
v_add_u32 v24, v24, 1                              // s59 = s[sgprWorkGroup0] / s58
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v25, v24, s58                         // s59 = s[sgprWorkGroup0] / s58
v_sub_u32 v26, s[sgprWorkGroup0], v25              // s59 = s[sgprWorkGroup0] / s58
v_readfirstlane_b32 s59, v24                       // quotient
v_readfirstlane_b32 s60, v26                       // remainder
s_waitcnt lgkmcnt(0)                               // wait for args to load
/* divmod(WG, WGMXCC) */
v_cvt_f64_u32 v[24:25], s58                        // s61 = s[sgprskGrid] / s58
v_rcp_f64 v[24:25], v[24:25]                       // s61 = s[sgprskGrid] / s58
v_cvt_f64_u32 v[26:27], s[sgprskGrid]              // s61 = s[sgprskGrid] / s58
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s61 = s[sgprskGrid] / s58
v_cvt_u32_f64 v24, v[24:25]                        // s61 = s[sgprskGrid] / s58
v_mul_lo_u32 v25, v24, s58                         // s61 = s[sgprskGrid] / s58
v_sub_u32 v26, s[sgprskGrid], v25                  // s61 = s[sgprskGrid] / s58
v_cmpx_ge_u32 exec, v26, s58                       // s61 = s[sgprskGrid] / s58
v_add_u32 v24, v24, 1                              // s61 = s[sgprskGrid] / s58
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v25, v24, s58                         // s61 = s[sgprskGrid] / s58
v_sub_u32 v26, s[sgprskGrid], v25                  // s61 = s[sgprskGrid] / s58
v_readfirstlane_b32 s61, v24                       // quotient
v_readfirstlane_b32 s62, v26                       // remainder
s_cmp_lt_u32 s60, s62
s_cselect_b32 s58, 0x1, 0x0                        // Select multiplier
s_cselect_b32 s63, 0x0, s62                        // Select remainder
s_add_u32 s61, s61, s58                            // Adjust multiplier
s_add_u32 s[sgprWorkGroup0], s59, s63
s_mul_i32 s58, s60, s61
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s17, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [24...136) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...96) to pool */
v_mov_b32 v26, MT0                                 // set MT0 into sgpr
v_mov_b32 v25, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v24, v26                             // v24 = ceil(v25 / v26)
v_rcp_iflag_f32 v24, v24                           // v24 = ceil(v25 / v26)
v_cvt_f32_u32 v27, v25                             // v24 = ceil(v25 / v26)
v_mul_f32 v24, v24, v27                            // v24 = ceil(v25 / v26)
v_cvt_u32_f32 v24, v24                             // v24 = ceil(v25 / v26)
v_mul_u32_u24 v27, v24, v26                        // v24 = ceil(v25 / v26)
v_sub_u32 v27, v25, v27                            // v24 = ceil(v25 / v26)
v_cmp_ne_u32 vcc, v27, 0                           // v24 = ceil(v25 / v26)
v_addc_co_u32 v24, vcc, v24, 0, vcc                // ceil
v_mov_b32 v26, MT1                                 // set MT1 into sgpr
v_mov_b32 v25, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v24     // set back to numWorkGroup0
v_cvt_f32_u32 v24, v26                             // v24 = ceil(v25 / v26)
v_rcp_iflag_f32 v24, v24                           // v24 = ceil(v25 / v26)
v_cvt_f32_u32 v27, v25                             // v24 = ceil(v25 / v26)
v_mul_f32 v24, v24, v27                            // v24 = ceil(v25 / v26)
v_cvt_u32_f32 v24, v24                             // v24 = ceil(v25 / v26)
v_mul_u32_u24 v27, v24, v26                        // v24 = ceil(v25 / v26)
v_sub_u32 v27, v25, v27                            // v24 = ceil(v25 / v26)
v_cmp_ne_u32 vcc, v27, 0                           // v24 = ceil(v25 / v26)
v_addc_co_u32 v24, vcc, v24, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v24     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 168
s_mul_i32 s64, s16, 4
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 224
s_mov_b32 s64, 0
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s10, 1
s_mov_b32 s65, 0
s_load_dwordx4 s[20:23], s[58:59], s64
s_cmpk_eq_u32 s16, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s20, 7                             // s62 = s20 / 128
s_and_b32 s60, 127, s20                            // s60 = s20 % 128
s_addc_u32 s62, s62, 0
s_mov_b32 s61, 0                                   // STATIC_DIV: divisor=192
s_mul_i32 s60, 682, s21                            // tmp1 = dividend * magic hi
s_lshl_b64 s[60:61], s[60:61], 16                  // left shift 16 bits
s_mul_i32 s63, s21, 43691                          // tmp0 = dividend * magic lo
s_add_u32 s60, s63, s60                            // add lo
s_addc_u32 s61, s61, 0                             // add hi
s_lshr_b64 s[60:61], s[60:61], 33                  // tmp0 = quotient
s_mul_i32 s61, s60, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s61, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s63, s60, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s22
s_add_u32 s65, s65, s62
s_cmp_lt_u32 s[sgprWorkGroup0], s65
s_cbranch_scc1 label_FOUND
s_add_u32 s64, s64, s11
s_load_dwordx4 s[20:23], s[58:59], s64
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, s16
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s20, 7                             // s62 = s20 / 128
s_and_b32 s60, 127, s20                            // s60 = s20 % 128
s_addc_u32 s62, s62, 0
s_mov_b32 s61, 0                                   // STATIC_DIV: divisor=192
s_mul_i32 s60, 682, s21                            // tmp1 = dividend * magic hi
s_lshl_b64 s[60:61], s[60:61], 16                  // left shift 16 bits
s_mul_i32 s63, s21, 43691                          // tmp0 = dividend * magic lo
s_add_u32 s60, s63, s60                            // add lo
s_addc_u32 s61, s61, 0                             // add hi
s_lshr_b64 s[60:61], s[60:61], 33                  // tmp0 = quotient
s_mul_i32 s61, s60, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s61, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s63, s60, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s22
s_add_u32 s65, s65, s62

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s59, s10, 1
s_sub_u32 s58, s65, s62
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s16, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 168
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx4 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 224
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx4 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 140 // 140
label_LoadExternalStructEnd:
/* init: add vgpr [24...136) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...96) to pool */
v_mov_b32 v26, MT0                                 // set MT0 into sgpr
v_mov_b32 v25, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v24, v26                             // v24 = ceil(v25 / v26)
v_rcp_iflag_f32 v24, v24                           // v24 = ceil(v25 / v26)
v_cvt_f32_u32 v27, v25                             // v24 = ceil(v25 / v26)
v_mul_f32 v24, v24, v27                            // v24 = ceil(v25 / v26)
v_cvt_u32_f32 v24, v24                             // v24 = ceil(v25 / v26)
v_mul_u32_u24 v27, v24, v26                        // v24 = ceil(v25 / v26)
v_sub_u32 v27, v25, v27                            // v24 = ceil(v25 / v26)
v_cmp_ne_u32 vcc, v27, 0                           // v24 = ceil(v25 / v26)
v_addc_co_u32 v24, vcc, v24, 0, vcc                // ceil
v_mov_b32 v26, MT1                                 // set MT1 into sgpr
v_mov_b32 v25, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v24     // set back to numWorkGroup0
v_cvt_f32_u32 v24, v26                             // v24 = ceil(v25 / v26)
v_rcp_iflag_f32 v24, v24                           // v24 = ceil(v25 / v26)
v_cvt_f32_u32 v27, v25                             // v24 = ceil(v25 / v26)
v_mul_f32 v24, v24, v27                            // v24 = ceil(v25 / v26)
v_cvt_u32_f32 v24, v24                             // v24 = ceil(v25 / v26)
v_mul_u32_u24 v27, v24, v26                        // v24 = ceil(v25 / v26)
v_sub_u32 v27, v25, v27                            // v24 = ceil(v25 / v26)
v_cmp_ne_u32 vcc, v27, 0                           // v24 = ceil(v25 / v26)
v_addc_co_u32 v24, vcc, v24, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v24     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 64
.set sgprSrdB, 68
.set sgprShadowLimitA, 58
.set sgprShadowLimitB, 72
.set sgprStaggerUIter, 74
.set sgprWrapUA, 75
.set sgprWrapUB, 77
.set sgprGlobalReadIncsA, 79
.set sgprGlobalReadIncsB, 80
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
s_mov_b32 s[sgprStreamKIdx], s[sgprWorkGroup0]     // Save original StreamK index
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitInit                  // Jump to single kernel init
v_cvt_f32_u32 v24, s[sgprSkSplit]                  // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_rcp_iflag_f32 v24, v24                           // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_f32_u32 v25, s[sgprStreamKIdx]               // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_f32 v24, v24, v25                            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_u32_f32 v24, v24                             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_u32_u24 v25, v24, s[sgprSkSplit]             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_sub_u32 v25, s[sgprStreamKIdx], v25              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cmpx_eq_u32 exec, v25, s[sgprSkSplit]            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_add_u32 v24, 1, v24                              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mov_b32 v25, 0                                   // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v25, s[sgprSkSplit]            // overflow happened in remainder
v_sub_u32 v24, v24, 1                              // quotient - 1
v_mul_u32_u24 v25, v24, s[sgprSkSplit]             // re-calculate remainder
v_sub_u32 v25, s[sgprStreamKIdx], v25              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s12, v24                       // quotient
v_readfirstlane_b32 s13, v25                       // remainder
s_mul_i32 s14, s[sgprSkSplit], s[sgprSKItersPerWG]
s_sub_u32 s14, s[sgprItersPerTile], s14            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprStreamKIter], s13, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s13, s14                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s14 // This WG does not have an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s13 // This WG has an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s12, s12, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Offset to correct tile
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s13                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mov_b32 s[sgprStreamKIterEnd], s[sgprTotalIters] // DP ending iteration (case: only DP work to do)
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s12, s[sgprTotalIters]                // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprStreamKIdx], s14              // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s12                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s13, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s14, s[sgprStreamKIterEnd] // Set end iter
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Cap ending iter at total SK iters
label_SK_InitDone:
s_cmp_lt_u32 s[sgprStreamKIter], s[sgprTotalIters] // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc0
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_NoBranch_T8JHFHKM7BO5OHXW:

/******************************************/
/* Persistent Loop Start                  */
/******************************************/
label_PersistentLoopStart:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
/* StreamK calculate tile idx and map to WG */
s_mul_hi_u32 s13, s[sgprStreamKIter], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s14, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s12, s[sgprStreamKIter], s14             // s_magic mul, div alg 2
s_add_u32 s12, s12, s13
s_and_b32 s14, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s12, s12, s14                           // sMagicDiv Alg 2
s_mul_i32 s13, s12, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s14, s13, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprStreamKIter], s13 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprStreamKIterEnd], s14 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s13 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s13, s[sgprStreamKIterEnd]               // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s15, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_sub_u32 s15, s[sgprTotalIters], s15              // Offset to first SK tile
s_mul_i32 s13, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s13, s13, s[sgprStreamKIter]             // Add DP shift
s_cmp_lt_u32 s13, s15                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s13, s14                                 // SK iterations shift
s_cmp_le_u32 s15, s[sgprStreamKIter]               // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mul_i32 s16, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s17, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s16, s16, s17                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s16 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s18, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s17, s[sgprStreamKIdx], s18              // StreamK starting iteration (case: before extra iters)
s_add_u32 s18, s17, s18                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s16                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s17, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s18, s[sgprStreamKIterEnd] // Set end iter
s_add_u32 s13, s[sgprStreamKIter], s15             // Offset to start of SK section
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s15 // Offset to start of SK section
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s[sgprTotalIters] // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprStreamKIter], s[sgprTotalIters] // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_S4FDBQ587JJL6NOU     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_KernelEnd, 4                  // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_KernelEnd
label_NoBranch_S4FDBQ587JJL6NOU:
label_SK_UpdateDone:
s_mov_b32 s[sgprStreamKIter], s13                  // Store current iteration
/* Map StreamK tile index to wg0/1/2 */
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v24, s13                             // TileID // nWG0*nWG1
v_rcp_iflag_f32 v24, v24                           // TileID // nWG0*nWG1
v_cvt_f32_u32 v25, s12                             // TileID // nWG0*nWG1
v_mul_f32 v24, v24, v25                            // TileID // nWG0*nWG1
v_cvt_u32_f32 v24, v24                             // TileID // nWG0*nWG1
v_mul_u32_u24 v25, v24, s13                        // TileID // nWG0*nWG1
v_sub_u32 v25, s12, v25                            // TileID // nWG0*nWG1
v_cmpx_eq_u32 exec, v25, s13                       // TileID // nWG0*nWG1
v_add_u32 v24, 1, v24                              // TileID // nWG0*nWG1
v_mov_b32 v25, 0                                   // TileID // nWG0*nWG1
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v25, s13                       // overflow happened in remainder
v_sub_u32 v24, v24, 1                              // quotient - 1
v_mul_u32_u24 v25, v24, s13                        // re-calculate remainder
v_sub_u32 v25, s12, v25                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v24         // quotient
v_readfirstlane_b32 s14, v25                       // remainder
v_cvt_f32_u32 v24, s[sgprNumWorkGroups0]           // TileID // nWG0
v_rcp_iflag_f32 v24, v24                           // TileID // nWG0
v_cvt_f32_u32 v25, s14                             // TileID // nWG0
v_mul_f32 v24, v24, v25                            // TileID // nWG0
v_cvt_u32_f32 v24, v24                             // TileID // nWG0
v_mul_u32_u24 v25, v24, s[sgprNumWorkGroups0]      // TileID // nWG0
v_sub_u32 v25, s14, v25                            // TileID // nWG0
v_cmpx_eq_u32 exec, v25, s[sgprNumWorkGroups0]     // TileID // nWG0
v_add_u32 v24, 1, v24                              // TileID // nWG0
v_mov_b32 v25, 0                                   // TileID // nWG0
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v25, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v24, v24, 1                              // quotient - 1
v_mul_u32_u24 v25, v24, s[sgprNumWorkGroups0]      // re-calculate remainder
v_sub_u32 v25, s14, v25                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v24         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v25         // remainder

v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck                  // branch if s[Alpha] != 0
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_UR8VN3A1SJCPC6PO     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_SK_CloseLoop, 4               // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_SK_CloseLoop
label_NoBranch_UR8VN3A1SJCPC6PO:
s_mov_b32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Skip iterations
label_SKAlphaCheck:
/* WGM Calculation */
s_mov_b32 s12, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s12, s12                            // Restore WGM
s_cmp_gt_i32 s12, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s12, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s12, s12                                 // abs(WGM)
v_cvt_f64_u32 v[24:25], s12                        // s13 = s[sgprWorkGroup0] / s12
v_rcp_f64 v[24:25], v[24:25]                       // s13 = s[sgprWorkGroup0] / s12
v_cvt_f64_u32 v[26:27], s[sgprWorkGroup0]          // s13 = s[sgprWorkGroup0] / s12
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s13 = s[sgprWorkGroup0] / s12
v_cvt_u32_f64 v24, v[24:25]                        // s13 = s[sgprWorkGroup0] / s12
v_mul_lo_u32 v25, v24, s12                         // s13 = s[sgprWorkGroup0] / s12
v_sub_u32 v26, s[sgprWorkGroup0], v25              // s13 = s[sgprWorkGroup0] / s12
v_cmpx_ge_u32 exec, v26, s12                       // s13 = s[sgprWorkGroup0] / s12
v_add_u32 v24, v24, 1                              // s13 = s[sgprWorkGroup0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v24                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup0], s16              // WorkGroup0=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s16, s16, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[24:25], s12                        // s14 = s[sgprNumWorkGroups0] / s12
v_rcp_f64 v[24:25], v[24:25]                       // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_f64_u32 v[26:27], s[sgprNumWorkGroups0]      // s14 = s[sgprNumWorkGroups0] / s12
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_u32_f64 v24, v[24:25]                        // s14 = s[sgprNumWorkGroups0] / s12
v_mul_lo_u32 v25, v24, s12                         // s14 = s[sgprNumWorkGroups0] / s12
v_sub_u32 v26, s[sgprNumWorkGroups0], v25          // s14 = s[sgprNumWorkGroups0] / s12
v_cmpx_ge_u32 exec, v26, s12                       // s14 = s[sgprNumWorkGroups0] / s12
v_add_u32 v24, v24, 1                              // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v24                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups0], s15          // NumWorkGroups0=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[24:25], s14                        // s[sgprWorkGroup1] = s16 / s14
v_rcp_f64 v[24:25], v[24:25]                       // s[sgprWorkGroup1] = s16 / s14
v_cvt_f64_u32 v[26:27], s16                        // s[sgprWorkGroup1] = s16 / s14
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s[sgprWorkGroup1] = s16 / s14
v_cvt_u32_f64 v24, v[24:25]                        // s[sgprWorkGroup1] = s16 / s14
v_mul_lo_u32 v25, v24, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v26, s16, v25                            // s[sgprWorkGroup1] = s16 / s14
v_cmpx_ge_u32 exec, v26, s14                       // s[sgprWorkGroup1] = s16 / s14
v_add_u32 v24, v24, 1                              // s[sgprWorkGroup1] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v25, v24, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v26, s16, v25                            // s[sgprWorkGroup1] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup1], v24         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v26         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s16, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s13 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s12, s12                                 // WGM
v_cvt_f64_u32 v[24:25], s12                        // s13 = s[sgprWorkGroup1] / s12
v_rcp_f64 v[24:25], v[24:25]                       // s13 = s[sgprWorkGroup1] / s12
v_cvt_f64_u32 v[26:27], s[sgprWorkGroup1]          // s13 = s[sgprWorkGroup1] / s12
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s13 = s[sgprWorkGroup1] / s12
v_cvt_u32_f64 v24, v[24:25]                        // s13 = s[sgprWorkGroup1] / s12
v_mul_lo_u32 v25, v24, s12                         // s13 = s[sgprWorkGroup1] / s12
v_sub_u32 v26, s[sgprWorkGroup1], v25              // s13 = s[sgprWorkGroup1] / s12
v_cmpx_ge_u32 exec, v26, s12                       // s13 = s[sgprWorkGroup1] / s12
v_add_u32 v24, v24, 1                              // s13 = s[sgprWorkGroup1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v24                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup1], s16              // WorkGroup1=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s16, s16, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[24:25], s12                        // s14 = s[sgprNumWorkGroups1] / s12
v_rcp_f64 v[24:25], v[24:25]                       // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_f64_u32 v[26:27], s[sgprNumWorkGroups1]      // s14 = s[sgprNumWorkGroups1] / s12
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_u32_f64 v24, v[24:25]                        // s14 = s[sgprNumWorkGroups1] / s12
v_mul_lo_u32 v25, v24, s12                         // s14 = s[sgprNumWorkGroups1] / s12
v_sub_u32 v26, s[sgprNumWorkGroups1], v25          // s14 = s[sgprNumWorkGroups1] / s12
v_cmpx_ge_u32 exec, v26, s12                       // s14 = s[sgprNumWorkGroups1] / s12
v_add_u32 v24, v24, 1                              // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v24                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups1], s15          // NumWorkGroups1=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[24:25], s14                        // s[sgprWorkGroup0] = s16 / s14
v_rcp_f64 v[24:25], v[24:25]                       // s[sgprWorkGroup0] = s16 / s14
v_cvt_f64_u32 v[26:27], s16                        // s[sgprWorkGroup0] = s16 / s14
v_mul_f64 v[24:25], v[24:25], v[26:27]             // s[sgprWorkGroup0] = s16 / s14
v_cvt_u32_f64 v24, v[24:25]                        // s[sgprWorkGroup0] = s16 / s14
v_mul_lo_u32 v25, v24, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v26, s16, v25                            // s[sgprWorkGroup0] = s16 / s14
v_cmpx_ge_u32 exec, v26, s14                       // s[sgprWorkGroup0] = s16 / s14
v_add_u32 v24, v24, 1                              // s[sgprWorkGroup0] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v25, v24, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v26, s16, v25                            // s[sgprWorkGroup0] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup0], v24         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v26         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s16, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s13 // wg1 += blockId * WGM
label_WGM:

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v25, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v24, 15, v25                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v24, 7, v24                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v24, 3, v24                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(8)
v_lshrrev_b32 v25, 4, v25                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v24, v25, 3, v24                    // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v26, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v25, 15, v26                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v25, 7, v25                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v26, 4, v26                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v25, v26, 3, v25                    // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v27, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v27, 3, v27                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v25, v27, 11, v25                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v26, 6, v[vgprSerial]                // 26 = Serial / 64
v_lshrrev_b32 v26, 2, v26                          // LSU offset: Get LSU wave_id
s_mov_b32 s12, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v26, s12, v26                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v26, v24, 0x1 // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v27, 11, v[vgprLocalReadAddrA]       // Final Offset: padding 32 per block 2048
v_lshl_add_u32 v[vgprLocalReadAddrA], v27, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 2048

/* local read addresses: final offsets b */
v_lshrrev_b32 v24, 6, v[vgprSerial]                // 24 = Serial / 64
v_lshrrev_b32 v24, 2, v24                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v24, s12, v24                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v24, v25, 0x1 // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v26, 8, v[vgprLocalReadAddrB]        // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v26, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x8200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v25 = A-unroll = serial%LVCA */
v_lshrrev_b32 v24, 4, v[vgprSerial]                // 24 = Serial / 16
v_and_b32 v25, 15, v[vgprSerial]                   // 25 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v25, 3, v25                          // v25 = v25 * 8
v_mov_b32 v28, v25                                 // copy for GlobalSplitU
/* LVCB = 16 */
/* v27 = B-unroll = serial%LVCB */
v_lshrrev_b32 v26, 4, v[vgprSerial]                // 26 = Serial / 16
v_and_b32 v27, 15, v[vgprSerial]                   // 27 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v27, 3, v27                          // v27 = v27 * 8
v_mov_b32 v29, v27                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v28 */
/* lwaUnrollAssignmentB = v29 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v24    // lwAL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v28, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v30, 11, v[vgprLocalWriteAddrA]      // padding 32 per block 2048
v_lshl_add_u32 v[vgprLocalWriteAddrA], v30, 5, v[vgprLocalWriteAddrA] // padding 32 per block 2048

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v26    // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v29, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v30, 8, v[vgprLocalWriteAddrB]       // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v30, 5, v[vgprLocalWriteAddrB] // padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x8200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=33280

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v24 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v26 */

/* global read addresses: unroll assignment a */
/* v25 */

/* global read addresses: unroll assignment b */
/* v27 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v30, v24                                 // groA0I_0
v_add_co_u32 v31, vcc, 16, v30                     // groA0I_1 += LSPA
v_add_co_u32 v32, vcc, 16, v31                     // groA0I_2 += LSPA
v_add_co_u32 v33, vcc, 16, v32                     // groA0I_3 += LSPA
v_add_co_u32 v34, vcc, 16, v33                     // groA0I_4 += LSPA
v_add_co_u32 v35, vcc, 16, v34                     // groA0I_5 += LSPA
v_add_co_u32 v36, vcc, 16, v35                     // groA0I_6 += LSPA
v_add_co_u32 v37, vcc, 16, v36                     // groA0I_7 += LSPA

// label_jzhou:
/* global read addresses: tile offsets b */
.if 0 //original
v_mov_b32 v38, v26                                 // groB1J_0
v_add_co_u32 v39, vcc, 16, v38                     // groB1J_1 += LSPB
v_add_co_u32 v40, vcc, 16, v39                     // groB1J_2 += LSPB
v_add_co_u32 v41, vcc, 16, v40                     // groB1J_3 += LSPB
v_add_co_u32 v42, vcc, 16, v41                     // groB1J_4 += LSPB
v_add_co_u32 v43, vcc, 16, v42                     // groB1J_5 += LSPB
v_add_co_u32 v44, vcc, 16, v43                     // groB1J_6 += LSPB
v_add_co_u32 v45, vcc, 16, v44                     // groB1J_7 += LSPB
v_add_co_u32 v46, vcc, 16, v45                     // groB1J_8 += LSPB
v_add_co_u32 v47, vcc, 16, v46                     // groB1J_9 += LSPB
v_add_co_u32 v48, vcc, 16, v47                     // groB1J_10 += LSPB
v_add_co_u32 v49, vcc, 16, v48                     // groB1J_11 += LSPB
.else 
v_lshlrev_b32 v38, 4, v26                           // groB1J_0 = v26 * 16
v_add_co_u32 v39, vcc, 256, v38                    // groB1J_1 = base + 256
v_add_co_u32 v40, vcc, 256, v39                    // groB1J_2 = base + 512
v_add_co_u32 v41, vcc, 256, v40                    // groB1J_3 = base + 768
v_add_co_u32 v42, vcc, 256, v41                    // groB1J_4 = base + 1024
v_add_co_u32 v43, vcc, 256, v42                    // groB1J_5 = base + 1280
v_add_co_u32 v44, vcc, 256, v43                    // groB1J_6 = base + 1536
v_add_co_u32 v45, vcc, 256, v44                    // groB1J_7 = base + 1792
v_add_co_u32 v46, vcc, 256, v45                    // groB1J_8 = base + 2048
v_add_co_u32 v47, vcc, 256, v46                    // groB1J_9 = base + 2304
v_add_co_u32 v48, vcc, 256, v47                    // groB1J_10 = base + 2560
v_add_co_u32 v49, vcc, 256, v48                    // groB1J_11 = base + 2816
.endif

/* global read addresses: unroll offsets a */
v_mov_b32 v50, v25                                 // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v51, v27                                 // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], DepthU    // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideAL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideAL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideAL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideAL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideA0I], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideA0I], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s13, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s15   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
// label_jzhou:
//s_mul_hi_u32 s15, s[sgprWorkGroup1], 192           // WorkGroup[01] * MT
//s_mul_i32 s14, s[sgprWorkGroup1], 192              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s[sgprWorkGroup1], 1           // WorkGroup[01] * 1
s_mul_i32 s14, s[sgprWorkGroup1], 1              // WorkGroup[01] * 1

s_mul_hi_u32 s15, s14, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], DepthU    // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideBL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideBL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideBL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideBL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideB1J], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideB1J], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s13, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s15   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0, 50, 30, 52 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1, 50, 31, 52 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2, 50, 32, 52 // gROA_0_0_2_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3, 50, 33, 52 // gROA_0_0_3_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+4, 50, 34, 52 // gROA_0_0_4_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+5, 50, 35, 52 // gROA_0_0_5_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+6, 50, 36, 52 // gROA_0_0_6_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+7, 50, 37, 52 // gROA_0_0_7_0
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0, 51, 38, 30 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+1, 51, 39, 30 // gROB_0_0_1_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+2, 51, 40, 30 // gROB_0_0_2_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+3, 51, 41, 30 // gROB_0_0_3_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+4, 51, 42, 30 // gROB_0_0_4_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+5, 51, 43, 30 // gROB_0_0_5_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+6, 51, 44, 30 // gROB_0_0_6_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+7, 51, 45, 30 // gROB_0_0_7_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+8, 51, 46, 30 // gROB_0_0_8_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+9, 51, 47, 30 // gROB_0_0_9_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+10, 51, 48, 30 // gROB_0_0_10_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+11, 51, 49, 30 // gROB_0_0_11_0
/* ============================================================= */

/* global read addresses: increments a */
s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeAGR  // incrA (unrollIdx)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR  // incrB (unrollIdx)
/* declare loop num iterations */
s_sub_u32 s[sgprLoopCounterL], s[sgprStreamKLocalEnd], s[sgprStreamKLocalStart] // StreamK loop counter = localEnd - localStart
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck2                 // branch if s[Alpha] != 0
s_mov_b32 s[sgprLoopCounterL], 0                   // Skip iterations
label_SKAlphaCheck2:
s_and_b32 s13, 127, s[sgprSizesSum+0]              // s13 = s[sgprSizesSum+0] % 128
s_cmp_eq_u32 s13, 0                                // numIterL == 0
s_cselect_b32 s12, 0, 1                            // check if size uses tail loop
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cselect_b32 s12, s12, 0                          // this WG runs tail loop
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], s12 // Adjust loop counter for tail loop
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s14, s[sgprStaggerU], 0x1f00
s_lshr_b32 s14, s14, 0x8
s_and_b32 s15, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s12, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s13, s12, s14                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s13           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s12, s12, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s13, s12, 1                              // staggerU mask
s_cmp_ge_u32 s12, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s13, 0          // set Mask
s_cmp_eq_u32 s15, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s15, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s12, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s15, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s15, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s12, s12, s13
s_add_u32 s12, s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s15, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s12 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s14 // shift by StaggerUStride
s_cmp_gt_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_dwordx4 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
buffer_load_dwordx4 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
buffer_load_dwordx4 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0

/* global read inc A loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s60, 1
s_mov_b32 s61, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s60, 1
s_mov_b32 s61, 2
label_BPEDone:
// label_jzhou:
//s_mul_i32 s84, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_i32 s84, 1, s[sgprWorkGroup1]              // <- wg1*1
s_mul_hi_u32 s83, s84, s[sgprStrideC1J]            // ScaleC s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideC1J]               // ScaleC s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s60                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s82    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s83   // add hi to SRD
s_mul_hi_u32 s83, s84, s[sgprStrideD1J]            // ScaleD s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideD1J]               // ScaleD s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s61                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s82    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s83   // add hi to SRD

s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s60                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // add hi to SRD
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s61                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi to SRD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s83, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s82, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s81, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s81, s81, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s84, s81, s[sgprStrideC1J]            // Free1
s_mul_i32 s81, s81, s[sgprStrideC1J]               // Free1
s_add_u32 s82, s82, s81                            // Free1
s_addc_u32 s83, s83, s84                           // Free1
s_sub_u32 s81, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s81, s81, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s84, s81, s[sgprStrideCK]             // Free2
s_mul_i32 s81, s81, s[sgprStrideCK]                // Free2
s_add_u32 s82, s82, s81                            // Free2
s_addc_u32 s83, s83, s84                           // Free2
s_lshl_b64 s[82:83], s[82:83], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...96) from pool */

/* initC: remove ValuA/B vgpr buffer [24...112) from pool */
v_accvgpr_write acc0, 0                            // initC
v_accvgpr_write acc1, 0                            // initC
v_accvgpr_write acc2, 0                            // initC
v_accvgpr_write acc3, 0                            // initC
v_accvgpr_write acc4, 0                            // initC
v_accvgpr_write acc5, 0                            // initC
v_accvgpr_write acc6, 0                            // initC
v_accvgpr_write acc7, 0                            // initC
v_accvgpr_write acc8, 0                            // initC
v_accvgpr_write acc9, 0                            // initC
v_accvgpr_write acc10, 0                           // initC
v_accvgpr_write acc11, 0                           // initC
v_accvgpr_write acc12, 0                           // initC
v_accvgpr_write acc13, 0                           // initC
v_accvgpr_write acc14, 0                           // initC
v_accvgpr_write acc15, 0                           // initC
v_accvgpr_write acc16, 0                           // initC
v_accvgpr_write acc17, 0                           // initC
v_accvgpr_write acc18, 0                           // initC
v_accvgpr_write acc19, 0                           // initC
v_accvgpr_write acc20, 0                           // initC
v_accvgpr_write acc21, 0                           // initC
v_accvgpr_write acc22, 0                           // initC
v_accvgpr_write acc23, 0                           // initC
v_accvgpr_write acc24, 0                           // initC
v_accvgpr_write acc25, 0                           // initC
v_accvgpr_write acc26, 0                           // initC
v_accvgpr_write acc27, 0                           // initC
v_accvgpr_write acc28, 0                           // initC
v_accvgpr_write acc29, 0                           // initC
v_accvgpr_write acc30, 0                           // initC
v_accvgpr_write acc31, 0                           // initC
v_accvgpr_write acc32, 0                           // initC
v_accvgpr_write acc33, 0                           // initC
v_accvgpr_write acc34, 0                           // initC
v_accvgpr_write acc35, 0                           // initC
v_accvgpr_write acc36, 0                           // initC
v_accvgpr_write acc37, 0                           // initC
v_accvgpr_write acc38, 0                           // initC
v_accvgpr_write acc39, 0                           // initC
v_accvgpr_write acc40, 0                           // initC
v_accvgpr_write acc41, 0                           // initC
v_accvgpr_write acc42, 0                           // initC
v_accvgpr_write acc43, 0                           // initC
v_accvgpr_write acc44, 0                           // initC
v_accvgpr_write acc45, 0                           // initC
v_accvgpr_write acc46, 0                           // initC
v_accvgpr_write acc47, 0                           // initC
v_accvgpr_write acc48, 0                           // initC
v_accvgpr_write acc49, 0                           // initC
v_accvgpr_write acc50, 0                           // initC
v_accvgpr_write acc51, 0                           // initC
v_accvgpr_write acc52, 0                           // initC
v_accvgpr_write acc53, 0                           // initC
v_accvgpr_write acc54, 0                           // initC
v_accvgpr_write acc55, 0                           // initC
v_accvgpr_write acc56, 0                           // initC
v_accvgpr_write acc57, 0                           // initC
v_accvgpr_write acc58, 0                           // initC
v_accvgpr_write acc59, 0                           // initC
v_accvgpr_write acc60, 0                           // initC
v_accvgpr_write acc61, 0                           // initC
v_accvgpr_write acc62, 0                           // initC
v_accvgpr_write acc63, 0                           // initC
v_accvgpr_write acc64, 0                           // initC
v_accvgpr_write acc65, 0                           // initC
v_accvgpr_write acc66, 0                           // initC
v_accvgpr_write acc67, 0                           // initC
v_accvgpr_write acc68, 0                           // initC
v_accvgpr_write acc69, 0                           // initC
v_accvgpr_write acc70, 0                           // initC
v_accvgpr_write acc71, 0                           // initC
v_accvgpr_write acc72, 0                           // initC
v_accvgpr_write acc73, 0                           // initC
v_accvgpr_write acc74, 0                           // initC
v_accvgpr_write acc75, 0                           // initC
v_accvgpr_write acc76, 0                           // initC
v_accvgpr_write acc77, 0                           // initC
v_accvgpr_write acc78, 0                           // initC
v_accvgpr_write acc79, 0                           // initC
v_accvgpr_write acc80, 0                           // initC
v_accvgpr_write acc81, 0                           // initC
v_accvgpr_write acc82, 0                           // initC
v_accvgpr_write acc83, 0                           // initC
v_accvgpr_write acc84, 0                           // initC
v_accvgpr_write acc85, 0                           // initC
v_accvgpr_write acc86, 0                           // initC
v_accvgpr_write acc87, 0                           // initC
v_accvgpr_write acc88, 0                           // initC
v_accvgpr_write acc89, 0                           // initC
v_accvgpr_write acc90, 0                           // initC
v_accvgpr_write acc91, 0                           // initC
v_accvgpr_write acc92, 0                           // initC
v_accvgpr_write acc93, 0                           // initC
v_accvgpr_write acc94, 0                           // initC
v_accvgpr_write acc95, 0                           // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_8S4L1KCK9VFC7AQU     // Only branch on scc1
s_getpc_b64 s[60:61]                               // addr of next instr
s_add_i32 s62, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s60, s60, s62                            // add target branch offset
s_addc_u32 s61, s61, 0                             // add high and carry
s_setpc_b64 s[60:61]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_8S4L1KCK9VFC7AQU:
s_waitcnt vmcnt(0)                                 // wait for global read
s_barrier                                          // For stream-k / persistent loop

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4160 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4160
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8320 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8320
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12480 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 12480
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:16640 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 16640
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:20800 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 20800
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:24960 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 24960
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:29120 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 29120

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+32:vgprG2LB+32+3] offset:36864 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 36864
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+36:vgprG2LB+36+3] offset:41472 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 41472
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+40:vgprG2LB+40+3] offset:46080 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 46080
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+44:vgprG2LB+44+3] offset:50688 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 50688

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2                      // PGR=2 but only 1 loop
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_dwordx4 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
buffer_load_dwordx4 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
buffer_load_dwordx4 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0
label_skipPGR2:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
// Skip force waitcnt0
s_barrier

/* local read prefetch a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:36928 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:24  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:25  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:26  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:27  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:28  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:29  */
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:30  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:31  */
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:32  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:33  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:34  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36992 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:38  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:42  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:46  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:47  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=3 */

/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:48  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:37056 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:64  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4160 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4160
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:65  */
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8320 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8320
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:66  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12480 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 12480
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:67  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:16640 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 16640
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:68  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:20800 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 20800
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:69  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:24960 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 24960
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:29120 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 29120
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:70  */
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:71  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 3 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:72  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=11 newLR=0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:73  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:74  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:75  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:76  */
buffer_load_dwordx4 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256
buffer_load_dwordx4 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:77  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+32:vgprG2LB+32+3] offset:36864 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 36864
buffer_load_dwordx4 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+36:vgprG2LB+36+3] offset:41472 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 41472
buffer_load_dwordx4 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+40:vgprG2LB+40+3] offset:46080 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 46080
buffer_load_dwordx4 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:80  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+44:vgprG2LB+44+3] offset:50688 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 50688
buffer_load_dwordx4 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:81  */

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:83  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:84  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:85  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:86  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:87  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:88  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:89  */
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:90  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:91  */
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:92  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:93  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:94  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:36928 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:24  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:25  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:26  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:27  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:28  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:29  */
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:30  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:31  */
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:32  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:33  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:34  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36992 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:38  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:42  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:46  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:47  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=3 */

/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:48  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:37056 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:64  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(19)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(18)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4160 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4160
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:65  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(17)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8320 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8320
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:66  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(16)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12480 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 12480
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:67  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:16640 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 16640
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:68  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:20800 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 20800
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:69  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:24960 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 24960
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:29120 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 29120
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:70  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:71  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 3 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:64, lwEndMfmaIndex:81  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:72  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=11 newLR=0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:73  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:74  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:75  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:76  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:77  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+32:vgprG2LB+32+3] offset:36864 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 36864
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+36:vgprG2LB+36+3] offset:41472 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 41472
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+40:vgprG2LB+40+3] offset:46080 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 46080
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:80  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+44:vgprG2LB+44+3] offset:50688 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 50688
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:81  */

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:83  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:84  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:85  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:86  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:87  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:88  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:89  */
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:90  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:91  */
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:92  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:93  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:94  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:71, lwEndMfmaIndex:71  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:576 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:832 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1088 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1344 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1600 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1856 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:36928 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:71, lwEndMfmaIndex:71  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:24  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:25  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:26  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:27  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:28  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:29  */
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:30  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:31  */
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:32  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:33  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:34  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36992 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:38  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:42  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:46  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:47  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=3 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:71, lwEndMfmaIndex:71  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:48  */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(1)                               // wait for prior local read local write old=0, new=1 newLW=0 newLR=1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
ds_read_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:704 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:960 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA] offset:1216 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA] offset:1472 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA] offset:1728 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA] offset:1984 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:37056 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:64  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:65  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:66  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:67  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:68  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:69  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:70  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:71  */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:71, lwEndMfmaIndex:71  */
/*  numMfmaForLR:12, syncPlrMfmaIndex:83  */
/*  mfmaIndex:72  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:73  */
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:74  */
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:75  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:76  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:77  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:80  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:81  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:83  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:84  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:85  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:86  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:87  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:88  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:89  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:90  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:91  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:92  */
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:93  */
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:94  */
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [24...112) to pool */

/* Tail: add address/G2L vgpr [112...192) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */

/* local write reset offsets b */
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (32,48,0) */
.set vgprG2LA_BASE, 24
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 56
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 127, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 128
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cmov_b32 s[sgprLoopCounterL], 0                  // This WG not completing tile
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s82, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s82, 0
s_cbranch_scc0 label_Negative_J5DQFVGFWLXU2DUR
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_DLSAQLEVYLOBCPNL
label_Negative_J5DQFVGFWLXU2DUR:
s_abs_i32 s82, s82
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s82, s82, 0xffffffff
s_xor_b32 s83, s83, 0xffffffff
s_add_u32 s82, s82, 0x1
s_addc_u32 s83, s83, 0
label_MultiplyDone_DLSAQLEVYLOBCPNL:
s_sub_u32 s82, s82, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s83, s83, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s82, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s82, 0
s_cbranch_scc0 label_Negative_LQI6BOBE0EY8XIP1
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_9N1QELR2XL4Z0HRB
label_Negative_LQI6BOBE0EY8XIP1:
s_abs_i32 s82, s82
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s82, s82, 0xffffffff
s_xor_b32 s83, s83, 0xffffffff
s_add_u32 s82, s82, 0x1
s_addc_u32 s83, s83, 0
label_MultiplyDone_9N1QELR2XL4Z0HRB:
s_sub_u32 s82, s82, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s83, s83, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* Update M0 for DTLDS */

/* Tail global read B */
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_dwordx4 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
buffer_load_dwordx4 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
buffer_load_dwordx4 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0

/* release sgprs that will not be used */
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF

/* find the last element location for a */
// Calculate SizeI % MacroTile0
s_mul_i32 s74, s[sgprWorkGroup0], 128              // Calculate the remaining dimension along I/J direction.
s_sub_u32 s74, s[sgprSizeI], s74                   // Calculate the remaining dimension along I/J direction.
s_mul_i32 s74, s74, 2                              // In bytes
s_and_b32 s78, s[sgprSizeL], 127                   // Calculate the remaining dimension along L direction.
s_lshr_b32 s86, s78, 0x7                           // Divided by lsc(128)
s_mul_hi_u32 s76, s74, s78                         // Calculate total number of valid elements.
s_mul_i32 s80, s74, s78                            // Calculate total number of valid elements.
s_cmp_gt_u32 s76, 0
s_cmov_b32 s80, 0xffffffff                         // If valid elements > max(U32), set the value to max
s_sub_u32 s78, s[sgprSizeI], 1                     // sLoadTileIdx starts from 0
// Calculate SizeI - 1 % MacroTile0
s_lshr_b32 s74, s78, 7                             // s74 = s78 / 128
s_and_b32 s74, 127, s78                            // s74 = s78 % 128
s_lshr_b32 s74, s74, 0x4                           // Divide lsp to get the load tile index
s_mul_i32 s74, s74, 1                              // Multiply nlc
s_add_i32 s74, s74, s86
s_and_b32 s78, 127, s[sgprSizesSum+0]              // s78 = s[sgprSizesSum+0] % 128
s_and_b32 s78, s78, 7                              // sLoadNum = (SizesSum+0 mod DU) & glvw
s_and_b32 s76, s78, 0x1

/* find the last element location for b */
// Calculate SizeJ % MacroTile1
s_mul_i32 s75, s[sgprWorkGroup1], 192              // Calculate the remaining dimension along I/J direction.
s_sub_u32 s75, s[sgprSizeJ], s75                   // Calculate the remaining dimension along I/J direction.
s_mul_i32 s75, s75, 2                              // In bytes
s_and_b32 s79, s[sgprSizeL], 127                   // Calculate the remaining dimension along L direction.
s_lshr_b32 s86, s79, 0x7                           // Divided by lsc(128)
s_mul_hi_u32 s77, s75, s79                         // Calculate total number of valid elements.
s_mul_i32 s81, s75, s79                            // Calculate total number of valid elements.
s_cmp_gt_u32 s77, 0
s_cmov_b32 s81, 0xffffffff                         // If valid elements > max(U32), set the value to max
s_sub_u32 s79, s[sgprSizeJ], 1                     // sLoadTileIdx starts from 0
// Calculate SizeJ - 1 % MacroTile1
s_mov_b32 s85, 0                                   // STATIC_DIV: divisor=192
s_mul_i32 s84, 682, s79                            // tmp1 = dividend * magic hi
s_lshl_b64 s[84:85], s[84:85], 16                  // left shift 16 bits
s_mul_i32 s75, s79, 43691                          // tmp0 = dividend * magic lo
s_add_u32 s84, s75, s84                            // add lo
s_addc_u32 s85, s85, 0                             // add hi
s_lshr_b64 s[84:85], s[84:85], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s75, s84                                 // quotient
s_mul_i32 s84, s75, 192                            // quotient*divisor
s_sub_u32 s75, s79, s84                            // rReg = dividend - quotient*divisor
s_lshr_b32 s75, s75, 0x4                           // Divide lsp to get the load tile index
s_mul_i32 s75, s75, 1                              // Multiply nlc
s_add_i32 s75, s75, s86
s_and_b32 s79, 127, s[sgprSizesSum+0]              // s79 = s[sgprSizesSum+0] % 128
s_and_b32 s79, s79, 7                              // sLoadNum = (SizesSum+0 mod DU) & glvw
s_and_b32 s77, s79, 0x1
s_mov_b32 s82, 0                                   // Set loop count = 0
s_mov_b32 s83, s77                                 // Backup and will be restored in label_CheckB_OOB

/* load single element for A */
label_LoadA:
s_cmp_eq_u32 s76, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_LoadB                         // Skip loading A
s_cmp_eq_u32 s74, 7
s_cbranch_scc1 label_LOAD_A7
s_cmp_eq_u32 s74, 6
s_cbranch_scc1 label_LOAD_A6
s_cmp_eq_u32 s74, 5
s_cbranch_scc1 label_LOAD_A5
s_cmp_eq_u32 s74, 4
s_cbranch_scc1 label_LOAD_A4
s_cmp_eq_u32 s74, 3
s_cbranch_scc1 label_LOAD_A3
s_cmp_eq_u32 s74, 2
s_cbranch_scc1 label_LOAD_A2
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_LOAD_A1
label_LOAD_A0:
label_LOAD_A0_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=0, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A0_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=0, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A0_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=0, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A0_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=0, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A1:
label_LOAD_A1_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=4, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A1_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=4, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A1_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=4, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A1_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=4, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A2:
label_LOAD_A2_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=8, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A2_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=8, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A2_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=8, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A2_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=8, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A3:
label_LOAD_A3_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=12, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A3_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=12, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A3_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=12, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A3_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=12, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A4:
label_LOAD_A4_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=16, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A4_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=16, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A4_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=16, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A4_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=16, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A5:
label_LOAD_A5_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=20, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A5_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=20, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A5_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=20, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A5_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=20, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A6:
label_LOAD_A6_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=24, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A6_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=24, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A6_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=24, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A6_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=24, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB
label_LOAD_A7:
label_LOAD_A7_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_LoadB
/* g2l=28, load component 0 */
buffer_load_short_d16 v104, v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
label_LOAD_A7_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_LoadB
/* g2l=28, load component 2 */
buffer_load_short_d16 v105, v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
label_LOAD_A7_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_LoadB
/* g2l=28, load component 4 */
buffer_load_short_d16 v106, v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
label_LOAD_A7_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_LoadB
/* g2l=28, load component 6 */
buffer_load_short_d16 v107, v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
s_branch label_LoadB

/* load single element for B */
label_LoadB:
s_cmp_eq_u32 s77, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_MergeA                        // Skip loading B
s_cmp_eq_u32 s75, 11
s_cbranch_scc1 label_LOAD_B11
s_cmp_eq_u32 s75, 10
s_cbranch_scc1 label_LOAD_B10
s_cmp_eq_u32 s75, 9
s_cbranch_scc1 label_LOAD_B9
s_cmp_eq_u32 s75, 8
s_cbranch_scc1 label_LOAD_B8
s_cmp_eq_u32 s75, 7
s_cbranch_scc1 label_LOAD_B7
s_cmp_eq_u32 s75, 6
s_cbranch_scc1 label_LOAD_B6
s_cmp_eq_u32 s75, 5
s_cbranch_scc1 label_LOAD_B5
s_cmp_eq_u32 s75, 4
s_cbranch_scc1 label_LOAD_B4
s_cmp_eq_u32 s75, 3
s_cbranch_scc1 label_LOAD_B3
s_cmp_eq_u32 s75, 2
s_cbranch_scc1 label_LOAD_B2
s_cmp_eq_u32 s75, 1
s_cbranch_scc1 label_LOAD_B1
label_LOAD_B0:
label_LOAD_B0_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=0, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B0_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=0, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B0_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=0, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B0_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=0, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B1:
label_LOAD_B1_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=4, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B1_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=4, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B1_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=4, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B1_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=4, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B2:
label_LOAD_B2_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=8, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B2_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=8, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B2_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=8, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B2_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=8, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B3:
label_LOAD_B3_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=12, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B3_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=12, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B3_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=12, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B3_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=12, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B4:
label_LOAD_B4_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=16, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B4_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=16, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B4_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=16, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B4_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=16, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B5:
label_LOAD_B5_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=20, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B5_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=20, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B5_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=20, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B5_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=20, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B6:
label_LOAD_B6_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=24, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B6_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=24, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B6_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=24, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B6_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=24, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B7:
label_LOAD_B7_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=28, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B7_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=28, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B7_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=28, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B7_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=28, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B8:
label_LOAD_B8_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=32, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B8_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=32, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B8_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=32, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B8_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=32, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B9:
label_LOAD_B9_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=36, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B9_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=36, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B9_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=36, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B9_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=36, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B10:
label_LOAD_B10_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=40, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B10_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=40, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B10_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=40, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B10_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=40, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA
label_LOAD_B11:
label_LOAD_B11_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_MergeA
/* g2l=44, load component 0 */
buffer_load_short_d16 v108, v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B11_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_MergeA
/* g2l=44, load component 2 */
buffer_load_short_d16 v109, v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B11_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_MergeA
/* g2l=44, load component 4 */
buffer_load_short_d16 v110, v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B11_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_MergeA
/* g2l=44, load component 6 */
buffer_load_short_d16 v111, v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeA

/* merge single element for A */
label_MergeA:
s_cmp_eq_u32 s76, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_MergeB                        // Skip mergeing A
s_cmp_eq_u32 s74, 7
s_cbranch_scc1 label_MERGE_A7
s_cmp_eq_u32 s74, 6
s_cbranch_scc1 label_MERGE_A6
s_cmp_eq_u32 s74, 5
s_cbranch_scc1 label_MERGE_A5
s_cmp_eq_u32 s74, 4
s_cbranch_scc1 label_MERGE_A4
s_cmp_eq_u32 s74, 3
s_cbranch_scc1 label_MERGE_A3
s_cmp_eq_u32 s74, 2
s_cbranch_scc1 label_MERGE_A2
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_MERGE_A1
label_MERGE_A0:
label_MERGE_A0_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v104    // HasEccHalf: pack
label_MERGE_A0_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v105    // HasEccHalf: pack
label_MERGE_A0_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v106    // HasEccHalf: pack
label_MERGE_A0_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v107    // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A1:
label_MERGE_A1_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v104    // HasEccHalf: pack
label_MERGE_A1_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v105    // HasEccHalf: pack
label_MERGE_A1_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v106    // HasEccHalf: pack
label_MERGE_A1_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v107    // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A2:
label_MERGE_A2_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v104    // HasEccHalf: pack
label_MERGE_A2_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v105    // HasEccHalf: pack
label_MERGE_A2_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v106    // HasEccHalf: pack
label_MERGE_A2_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v107    // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A3:
label_MERGE_A3_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+12+0], v[vgprG2LA+12+0], v104  // HasEccHalf: pack
label_MERGE_A3_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+12+1], v[vgprG2LA+12+1], v105  // HasEccHalf: pack
label_MERGE_A3_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+12+2], v[vgprG2LA+12+2], v106  // HasEccHalf: pack
label_MERGE_A3_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+12+3], v[vgprG2LA+12+3], v107  // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A4:
label_MERGE_A4_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+16+0], v[vgprG2LA+16+0], v104  // HasEccHalf: pack
label_MERGE_A4_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+16+1], v[vgprG2LA+16+1], v105  // HasEccHalf: pack
label_MERGE_A4_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+16+2], v[vgprG2LA+16+2], v106  // HasEccHalf: pack
label_MERGE_A4_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+16+3], v[vgprG2LA+16+3], v107  // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A5:
label_MERGE_A5_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+20+0], v[vgprG2LA+20+0], v104  // HasEccHalf: pack
label_MERGE_A5_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+20+1], v[vgprG2LA+20+1], v105  // HasEccHalf: pack
label_MERGE_A5_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+20+2], v[vgprG2LA+20+2], v106  // HasEccHalf: pack
label_MERGE_A5_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+20+3], v[vgprG2LA+20+3], v107  // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A6:
label_MERGE_A6_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+24+0], v[vgprG2LA+24+0], v104  // HasEccHalf: pack
label_MERGE_A6_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+24+1], v[vgprG2LA+24+1], v105  // HasEccHalf: pack
label_MERGE_A6_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+24+2], v[vgprG2LA+24+2], v106  // HasEccHalf: pack
label_MERGE_A6_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+24+3], v[vgprG2LA+24+3], v107  // HasEccHalf: pack
s_branch label_MergeB
label_MERGE_A7:
label_MERGE_A7_K1:
s_cmp_ge_u32 s78, 1
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+28+0], v[vgprG2LA+28+0], v104  // HasEccHalf: pack
label_MERGE_A7_K3:
s_cmp_ge_u32 s78, 3
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+28+1], v[vgprG2LA+28+1], v105  // HasEccHalf: pack
label_MERGE_A7_K5:
s_cmp_ge_u32 s78, 5
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+28+2], v[vgprG2LA+28+2], v106  // HasEccHalf: pack
label_MERGE_A7_K7:
s_cmp_ge_u32 s78, 7
s_cbranch_scc0 label_MergeB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+28+3], v[vgprG2LA+28+3], v107  // HasEccHalf: pack
s_branch label_MergeB

/* merge single element for B */
label_MergeB:
s_cmp_eq_u32 s77, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_CheckOtherLoadA               // Skip mergeing B
s_cmp_eq_u32 s75, 11
s_cbranch_scc1 label_MERGE_B11
s_cmp_eq_u32 s75, 10
s_cbranch_scc1 label_MERGE_B10
s_cmp_eq_u32 s75, 9
s_cbranch_scc1 label_MERGE_B9
s_cmp_eq_u32 s75, 8
s_cbranch_scc1 label_MERGE_B8
s_cmp_eq_u32 s75, 7
s_cbranch_scc1 label_MERGE_B7
s_cmp_eq_u32 s75, 6
s_cbranch_scc1 label_MERGE_B6
s_cmp_eq_u32 s75, 5
s_cbranch_scc1 label_MERGE_B5
s_cmp_eq_u32 s75, 4
s_cbranch_scc1 label_MERGE_B4
s_cmp_eq_u32 s75, 3
s_cbranch_scc1 label_MERGE_B3
s_cmp_eq_u32 s75, 2
s_cbranch_scc1 label_MERGE_B2
s_cmp_eq_u32 s75, 1
s_cbranch_scc1 label_MERGE_B1
label_MERGE_B0:
label_MERGE_B0_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v108    // HasEccHalf: pack
label_MERGE_B0_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v109    // HasEccHalf: pack
label_MERGE_B0_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v110    // HasEccHalf: pack
label_MERGE_B0_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v111    // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B1:
label_MERGE_B1_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v108    // HasEccHalf: pack
label_MERGE_B1_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v109    // HasEccHalf: pack
label_MERGE_B1_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v110    // HasEccHalf: pack
label_MERGE_B1_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v111    // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B2:
label_MERGE_B2_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v108    // HasEccHalf: pack
label_MERGE_B2_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v109    // HasEccHalf: pack
label_MERGE_B2_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v110    // HasEccHalf: pack
label_MERGE_B2_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v111    // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B3:
label_MERGE_B3_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v108  // HasEccHalf: pack
label_MERGE_B3_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v109  // HasEccHalf: pack
label_MERGE_B3_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v110  // HasEccHalf: pack
label_MERGE_B3_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B4:
label_MERGE_B4_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+16+0], v[vgprG2LB+16+0], v108  // HasEccHalf: pack
label_MERGE_B4_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+16+1], v[vgprG2LB+16+1], v109  // HasEccHalf: pack
label_MERGE_B4_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+16+2], v[vgprG2LB+16+2], v110  // HasEccHalf: pack
label_MERGE_B4_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+16+3], v[vgprG2LB+16+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B5:
label_MERGE_B5_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+20+0], v[vgprG2LB+20+0], v108  // HasEccHalf: pack
label_MERGE_B5_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+20+1], v[vgprG2LB+20+1], v109  // HasEccHalf: pack
label_MERGE_B5_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+20+2], v[vgprG2LB+20+2], v110  // HasEccHalf: pack
label_MERGE_B5_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+20+3], v[vgprG2LB+20+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B6:
label_MERGE_B6_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+24+0], v[vgprG2LB+24+0], v108  // HasEccHalf: pack
label_MERGE_B6_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+24+1], v[vgprG2LB+24+1], v109  // HasEccHalf: pack
label_MERGE_B6_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+24+2], v[vgprG2LB+24+2], v110  // HasEccHalf: pack
label_MERGE_B6_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+24+3], v[vgprG2LB+24+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B7:
label_MERGE_B7_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+28+0], v[vgprG2LB+28+0], v108  // HasEccHalf: pack
label_MERGE_B7_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+28+1], v[vgprG2LB+28+1], v109  // HasEccHalf: pack
label_MERGE_B7_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+28+2], v[vgprG2LB+28+2], v110  // HasEccHalf: pack
label_MERGE_B7_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+28+3], v[vgprG2LB+28+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B8:
label_MERGE_B8_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+32+0], v[vgprG2LB+32+0], v108  // HasEccHalf: pack
label_MERGE_B8_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+32+1], v[vgprG2LB+32+1], v109  // HasEccHalf: pack
label_MERGE_B8_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+32+2], v[vgprG2LB+32+2], v110  // HasEccHalf: pack
label_MERGE_B8_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+32+3], v[vgprG2LB+32+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B9:
label_MERGE_B9_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+36+0], v[vgprG2LB+36+0], v108  // HasEccHalf: pack
label_MERGE_B9_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+36+1], v[vgprG2LB+36+1], v109  // HasEccHalf: pack
label_MERGE_B9_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+36+2], v[vgprG2LB+36+2], v110  // HasEccHalf: pack
label_MERGE_B9_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+36+3], v[vgprG2LB+36+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B10:
label_MERGE_B10_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+40+0], v[vgprG2LB+40+0], v108  // HasEccHalf: pack
label_MERGE_B10_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+40+1], v[vgprG2LB+40+1], v109  // HasEccHalf: pack
label_MERGE_B10_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+40+2], v[vgprG2LB+40+2], v110  // HasEccHalf: pack
label_MERGE_B10_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+40+3], v[vgprG2LB+40+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA
label_MERGE_B11:
label_MERGE_B11_K1:
s_cmp_ge_u32 s79, 1
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+44+0], v[vgprG2LB+44+0], v108  // HasEccHalf: pack
label_MERGE_B11_K3:
s_cmp_ge_u32 s79, 3
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+44+1], v[vgprG2LB+44+1], v109  // HasEccHalf: pack
label_MERGE_B11_K5:
s_cmp_ge_u32 s79, 5
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+44+2], v[vgprG2LB+44+2], v110  // HasEccHalf: pack
label_MERGE_B11_K7:
s_cmp_ge_u32 s79, 7
s_cbranch_scc0 label_CheckOtherLoadA
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+44+3], v[vgprG2LB+44+3], v111  // HasEccHalf: pack
s_branch label_CheckOtherLoadA

/* reload loop for a: check if there's other load range need to be reloaded */
label_CheckOtherLoadA:
s_cmp_eq_u32 s76, 0                                // Noneed to load single element fo A?
s_cbranch_scc1 label_CheckOtherLoadB
s_add_u32 s82, s82, 1
s_cmp_eq_u32 s82, 8                                // Have reloaded all subtiles?
s_cmov_b32 s82, 0                                  // Reset loop count
s_cbranch_scc1 label_CheckOtherLoadB
s_sub_i32 s74, s74, 1                              // Check the upper subtile
s_cmp_lt_i32 s74, 0
s_cselect_b32 s84, 8, 0                            // Back to the last subtile
s_add_i32 s74, s74, s84                            // If currently reload the first subtile,                                   check the last subtile next.
s_cmp_eq_u32 s74, 7
s_cbranch_scc1 label_A7
s_cmp_eq_u32 s74, 6
s_cbranch_scc1 label_A6
s_cmp_eq_u32 s74, 5
s_cbranch_scc1 label_A5
s_cmp_eq_u32 s74, 4
s_cbranch_scc1 label_A4
s_cmp_eq_u32 s74, 3
s_cbranch_scc1 label_A3
s_cmp_eq_u32 s74, 2
s_cbranch_scc1 label_A2
s_cmp_eq_u32 s74, 1
s_cbranch_scc1 label_A1
label_A0:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+0]
s_branch label_CheckAddrA
label_A1:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+1]
s_branch label_CheckAddrA
label_A2:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+2]
s_branch label_CheckAddrA
label_A3:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+3]
s_branch label_CheckAddrA
label_A4:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+4]
s_branch label_CheckAddrA
label_A5:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+5]
s_branch label_CheckAddrA
label_A6:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+6]
s_branch label_CheckAddrA
label_A7:
v_mov_b32 v104, v[vgprGlobalReadOffsetA+7]
label_CheckAddrA:
v_sub_u32 v104, v104, 16                           // sub prepad
v_add_u32 v105, v104, 15                           // Calculate load range per thread
v_cmp_lt_i32 s[84:85], v104, s80                   // If loading start address < total valid bytes?
v_cmp_ge_i32 s[86:87], v105, s80                   // If loading end address >= total valid bytes?
s_and_b32 s84, s84, s86                            // Find threads which access the last element
s_and_b32 s85, s85, s87                            // Find thread that access the last element
s_add_u32 s84, s84, s85                            // Find thread that access the last element
s_cmp_lg_u32 s84, 0                                // Have threads access the last element?
s_cmov_b32 s77, 0                                  // Skip reload B temporarily
s_cselect_b32 s82, s82, 0                          // Reset loop count if needed
s_cbranch_scc1 label_LoadA                         // Reload A

/* reload loop for b: check if there's other load range need to be reloaded */
label_CheckOtherLoadB:
s_mov_b32 s76, 0                                   // Force to skip reload A
s_cmp_eq_u32 s82, 0                                // Loop start?
label_CheckB_OOB:
s_cmov_b32 s77, s83                                // Restore sReloadFlagB for B
s_cmp_eq_u32 s77, 0                                // Noneed to load single element for B?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_add_u32 s82, s82, 1
s_cmp_eq_u32 s82, 12                               // Have reloaded all subtiles?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_sub_i32 s75, s75, 1                              // Check the upper subtile
s_cmp_lt_i32 s75, 0
s_cselect_b32 s84, 12, 0                           // Back to the last subtile
s_add_i32 s75, s75, s84                            // If currently reload the first subtile,                                   check the last subtile next.
s_cmp_eq_u32 s75, 11
s_cbranch_scc1 label_B11
s_cmp_eq_u32 s75, 10
s_cbranch_scc1 label_B10
s_cmp_eq_u32 s75, 9
s_cbranch_scc1 label_B9
s_cmp_eq_u32 s75, 8
s_cbranch_scc1 label_B8
s_cmp_eq_u32 s75, 7
s_cbranch_scc1 label_B7
s_cmp_eq_u32 s75, 6
s_cbranch_scc1 label_B6
s_cmp_eq_u32 s75, 5
s_cbranch_scc1 label_B5
s_cmp_eq_u32 s75, 4
s_cbranch_scc1 label_B4
s_cmp_eq_u32 s75, 3
s_cbranch_scc1 label_B3
s_cmp_eq_u32 s75, 2
s_cbranch_scc1 label_B2
s_cmp_eq_u32 s75, 1
s_cbranch_scc1 label_B1
label_B0:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+0]
s_branch label_CheckAddrB
label_B1:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+1]
s_branch label_CheckAddrB
label_B2:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+2]
s_branch label_CheckAddrB
label_B3:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+3]
s_branch label_CheckAddrB
label_B4:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+4]
s_branch label_CheckAddrB
label_B5:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+5]
s_branch label_CheckAddrB
label_B6:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+6]
s_branch label_CheckAddrB
label_B7:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+7]
s_branch label_CheckAddrB
label_B8:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+8]
s_branch label_CheckAddrB
label_B9:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+9]
s_branch label_CheckAddrB
label_B10:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+10]
s_branch label_CheckAddrB
label_B11:
v_mov_b32 v104, v[vgprGlobalReadOffsetB+11]
label_CheckAddrB:
v_sub_u32 v104, v104, 16                           // sub prepad
v_add_u32 v105, v104, 15                           // Calculate load range per thread
v_cmp_lt_i32 s[84:85], v104, s81                   // If loading start address < total valid bytes?
v_cmp_ge_i32 s[86:87], v105, s81                   // If loading end address >= total valid bytes?
s_and_b32 s84, s84, s86                            // Find threads which access the last element
s_and_b32 s85, s85, s87                            // Find thread that access the last element
s_add_u32 s84, s84, s85                            // Find thread that access the last element
s_cmp_lg_u32 s84, 0                                // Have threads access the last element?
s_cbranch_scc1 label_LoadB                         // Reload B

/* global read for tail done */
label_TailGlobalLoadEnd:
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4160 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4160
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8320 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8320
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12480 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 12480
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:16640 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 16640
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:20800 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 20800
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:24960 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 24960
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:29120 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 29120

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+32:vgprG2LB+32+3] offset:36864 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 36864
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+36:vgprG2LB+36+3] offset:41472 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 41472
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+40:vgprG2LB+40+3] offset:46080 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 46080
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+44:vgprG2LB+44+3] offset:50688 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 50688

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 24
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuB_X0_I0_BASE, 88
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+12

/* Tail: local read reset offsets a */

/* Tail: local read reset offsets b */

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
s_mov_b32 s74, 64                                  // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s74, v[vgprLocalReadAddrA+0] // lrA += 64 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s74, v[vgprLocalReadAddrB+0] // lrB += 64 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v112, 63, v[vgprSerial]                  // v112 = v[vgprSerial] % 64
v_lshrrev_b32 v112, 4, v112                        // 112 = 112 / 16
v_lshlrev_b32 v112, 3, v112                        // v112 = v112 * 8
v_add_u32 v113, v112, 0
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+0], v[vgprValuA_X0_I0+8+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+0], v[vgprValuA_X0_I0+12+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+0], v[vgprValuA_X0_I0+16+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+0], v[vgprValuA_X0_I0+20+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+0], v[vgprValuA_X0_I0+24+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+0], v[vgprValuA_X0_I0+28+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+8+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+12+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+16+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+20+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+24+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+28+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+2], v[vgprValuA_X0_I0+4+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+2], v[vgprValuA_X0_I0+8+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+2], v[vgprValuA_X0_I0+12+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+2], v[vgprValuA_X0_I0+16+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+2], v[vgprValuA_X0_I0+20+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+2], v[vgprValuA_X0_I0+24+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+2], v[vgprValuA_X0_I0+28+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+3], v[vgprValuA_X0_I0+16+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+3], v[vgprValuA_X0_I0+20+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+3], v[vgprValuA_X0_I0+24+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+3], v[vgprValuA_X0_I0+28+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_and_b32 v112, 63, v[vgprSerial]                  // v112 = v[vgprSerial] % 64
v_lshrrev_b32 v112, 4, v112                        // 112 = 112 / 16
v_lshlrev_b32 v112, 3, v112                        // v112 = v112 * 8
v_add_u32 v113, v112, 0
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], 0, s[74:75] // set 0 if K_idx >= sizeL
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+2], v[vgprValuB_X0_I0+4+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+3], v[vgprValuB_X0_I0+4+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], 0, s[74:75] // set 0 if K_idx >= sizeL
s_and_b32 s76, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s76, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ // skip mask
s_and_b32 s76, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s76, 8, s76                              // use shift to fill 0 for outside element
s_lshl_b32 s76, s76, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+4+0+0+0:vgprValuA_X0_I0+4+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+2], v[vgprValuA_X0_I0+4+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+8+0+0+0:vgprValuA_X0_I0+8+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+8+0+0+2:vgprValuA_X0_I0+8+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+0], v[vgprValuA_X0_I0+8+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+8+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+2], v[vgprValuA_X0_I0+8+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+12+0+0+0:vgprValuA_X0_I0+12+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+12+0+0+2:vgprValuA_X0_I0+12+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+0], v[vgprValuA_X0_I0+12+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+12+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+2], v[vgprValuA_X0_I0+12+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+16+0+0+0:vgprValuA_X0_I0+16+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+16+0+0+2:vgprValuA_X0_I0+16+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+0], v[vgprValuA_X0_I0+16+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+16+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+2], v[vgprValuA_X0_I0+16+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0+3], v[vgprValuA_X0_I0+16+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+20+0+0+0:vgprValuA_X0_I0+20+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+20+0+0+2:vgprValuA_X0_I0+20+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+0], v[vgprValuA_X0_I0+20+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+20+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+2], v[vgprValuA_X0_I0+20+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0+3], v[vgprValuA_X0_I0+20+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+24+0+0+0:vgprValuA_X0_I0+24+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+24+0+0+2:vgprValuA_X0_I0+24+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+0], v[vgprValuA_X0_I0+24+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+24+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+2], v[vgprValuA_X0_I0+24+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0+3], v[vgprValuA_X0_I0+24+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuA_X0_I0+28+0+0+0:vgprValuA_X0_I0+28+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuA_X0_I0+28+0+0+2:vgprValuA_X0_I0+28+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+0], v[vgprValuA_X0_I0+28+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+28+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+2], v[vgprValuA_X0_I0+28+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0+3], v[vgprValuA_X0_I0+28+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuB_X0_I0+4+0+0+0:vgprValuB_X0_I0+4+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+2], v[vgprValuB_X0_I0+4+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+3], v[vgprValuB_X0_I0+4+0+0+3], v117, s[74:75]
v_lshlrev_b64 v[114:115], s76, v[vgprValuB_X0_I0+8+0+0+0:vgprValuB_X0_I0+8+0+0+0+1]
v_lshlrev_b64 v[116:117], s76, v[vgprValuB_X0_I0+8+0+0+2:vgprValuB_X0_I0+8+0+0+2+1]
v_add_u32 v113, v112, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], v114, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], v115, s[74:75]
v_add_u32 v113, v113, 4                            // add part of K
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], v116, s[74:75]
v_cmp_ge_i32 s[74:75], v113, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], v117, s[74:75]
label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ:
s_nop 1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[64:67] // left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[68:71] // left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[72:75] // left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[76:79] // left value = acc[76+0:79+0]
v_mfma_f32_16x16x32_bf16 acc[80:83], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+3], acc[80:83] // left value = acc[80+0:83+0]
v_mfma_f32_16x16x32_bf16 acc[84:87], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+3], acc[84:87] // left value = acc[84+0:87+0]
v_mfma_f32_16x16x32_bf16 acc[88:91], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+3], acc[88:91] // left value = acc[88+0:91+0]
v_mfma_f32_16x16x32_bf16 acc[92:95], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+3], acc[92:95] // left value = acc[92+0:95+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
s_mov_b32 s74, 2                                   // tailloop lds offset
s_mul_i32 s74, s[sgprOrigLoopCounter], s74         // scale by mul
v_sub_u32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], s74 // remove lro damage
s_mov_b32 s74, 2                                   // tailloop lds offset
s_mul_i32 s74, s[sgprOrigLoopCounter], s74         // scale by mul
v_sub_u32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], s74 // remove lro damage
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF

/* Tail: add MISC Vgpr [0...24) to pool */
label_Summation_End_QWMA7J3AUDGL0X23:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 64
.set sgprAddressBias, 66
.set sgprBiasType, 68
.set sgprBiasStride, 69
.set sgpractivationAlpha, 70
.set sgpractivationBeta, 71
.set sgprActivationType, 72
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalEpilogueStruct    // branch if ArgType == 2
s_load_dwordx8 s[64:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 132 // 132
s_load_dword s72, s[sgprKernArgAddress:sgprKernArgAddress+1], 164 // 164
s_branch label_LoadExternalEpilogueStructEnd
label_LoadExternalEpilogueStruct:
s_load_dwordx4 s[64:67], s[sgprKernArgAddress:sgprKernArgAddress+1], 188 // 188
s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 204 // 204
s_load_dwordx2 s[70:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 228 // 228
s_load_dword s72, s[sgprKernArgAddress:sgprKernArgAddress+1], 236 // 236
label_LoadExternalEpilogueStructEnd:
.set sgprSrdScaleAlphaVec, 76
.set sgprSrdBias, 80

/* Mapping of Acc register -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 0, v4                            // 5 = 4 / 1
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // 1 = 1 / 16
v_lshlrev_b32 v1, 2, v1                            // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
// label_jzhou:
v_lshrrev_b32 v3, 4, v[vgprSerial]                 // 1 = thread_id / 16     # 0 到 16
v_lshlrev_b32 v3, 6, v3                            // thread0 * 64 (改为乘以64)
v_mul_lo_u32 v3, v3, s[sgprStrideD1J]              //  offset 1

v_and_b32 v0, 0, v4                                // v0 = v4 % 1
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 3                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 192, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 36 bytes of kern args.

.set sgprAddressScaleAlphaVec, 64
.set sgprSrdScaleAlphaVec, 76
s_mov_b64 s[sgprSrdScaleAlphaVec+0:sgprSrdScaleAlphaVec+0+1], s[sgprAddressScaleAlphaVec+0:sgprAddressScaleAlphaVec+0+1] // init SRD base address
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVecAddrValid        // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVecAddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE
s_add_u32 s8, s[sgprWorkGroup2], 0x1
s_mul_i32 s8, s[sgprBiasStride], s8                // stride * (wg+1)
s_cmp_eq_u32 s8, 0                                 // bias stride = 0?
s_cselect_b32 s8, s[sgprSizeI], s8
s_mov_b64 s[sgprSrdBias+0:sgprSrdBias+0+1], s[sgprAddressBias+0:sgprAddressBias+0+1] // init SRD base address
s_mov_b32 s[sgprSrdBias+3], Srd127_96              // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressBias:sgprAddressBias+1], 0 // s[AddressBias] == 0 ?
s_cbranch_scc0 label_BiasAddrValid                 // branch if s[AddressBias] != 0
s_mov_b32 s[sgprSrdBias+2], 0
s_branch label_BiasAddrValid_End
label_BiasAddrValid:
s_mov_b32 s[sgprSrdBias+2], s8
label_BiasAddrValid_End:

label_Load_Biasf32_0:
s_cmpk_lg_u32 s[sgprBiasType], 0                   // BiasType != 0
s_cbranch_scc1 label_Load_Biasbf16_0               // Branch if true

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v8, s8, v[vgprSerial]                    // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s[sgprSrdBias+2], 0x4, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s8, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v6, s8, v8                               // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v6, 0x2, v6                          // Global bias address scaled by BPE
v_lshlrev_b32 v7, 0x2, v8                          // Global scaleAlpha address scaled by BPE
s_mul_i32 s8, 192, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v8, s8, v[vgprSerial]                    // coord 1 = wgp1 * MT1 + thread offset
buffer_load_dword v4, v6, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // Load Bias
buffer_load_dword v5, v7, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v8, 0x2, v[vgprSerial]               // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
s_waitcnt vmcnt(1)                                 // wait for global load
ds_write_b32 v8, v4 offset:0                       // store bias
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v5, 1.0, v5, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v8, v5 offset:1024                    // store scaleAlpha
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Biasbf16_0:
s_cmpk_lg_u32 s[sgprBiasType], 7                   // BiasType != 7
s_cbranch_scc1 label_Load_Bias_End                 // Branch if true

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v8, s8, v[vgprSerial]                    // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s[sgprSrdBias+2], 0x2, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s8, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v6, s8, v8                               // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v6, 0x1, v6                          // Global bias address scaled by BPE
v_lshlrev_b32 v7, 0x2, v8                          // Global scaleAlpha address scaled by BPE
s_mul_i32 s8, 192, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v8, s8, v[vgprSerial]                    // coord 1 = wgp1 * MT1 + thread offset
buffer_load_short_d16 v4, v6, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // Load Bias
buffer_load_dword v5, v7, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v8, 0x2, v[vgprSerial]               // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
s_waitcnt vmcnt(1)                                 // wait for global load
v_cvt_f32_bf16 v4, v4 src0_sel:WORD_0              // cvt bf16 to f32
ds_write_b32 v8, v4 offset:0                       // store bias
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v5, 1.0, v5, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v8, v5 offset:1024                    // store scaleAlpha
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Bias_End:
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_2G3LC8VCGIZD1EUX     // Only branch on scc0
s_getpc_b64 s[84:85]                               // addr of next instr
s_add_i32 s86, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s84, s84, s86                            // add target branch offset
s_addc_u32 s85, s85, 0                             // add high and carry
s_setpc_b64 s[84:85]                               // branch to label_SK_Partials_1
label_SK_Partials_1:
label_NoBranch_2G3LC8VCGIZD1EUX:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s8, s[sgprStreamKIdx], 1                 // input partial tile index
s_mul_hi_u32 s74, s[sgprStreamKIterEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s75, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s73, s[sgprStreamKIterEnd], s75          // s_magic mul, div alg 2
s_add_u32 s73, s73, s74
s_and_b32 s75, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s73, s73, s75                           // sMagicDiv Alg 2
s_mul_i32 s73, s73, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprStreamKIterEnd], s73           // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s73, s8, 2                              // flag offset based on CTA index
s_load_dword s75, s[sgprAddressFlags:sgprAddressFlags+1], s73 glc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s75, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s75, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s75, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
s_store_dword s75, s[sgprAddressFlags:sgprAddressFlags+1], s73 glc // reset flag
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
s_mov_b64 s[sgprSrdWS+0:sgprSrdWS+0+1], s[sgprAddressWS+0:sgprAddressWS+0+1] // init SRD base address
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96                // Set bits 127_96 in post-loop SRD

s_mul_i32 s58, 0x18000, s8                         // Offset to correct partials tile
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s58      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], 0       // add hi to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw8); (0,0,1,0:vw8); (0,0,2,0:vw8); (0,0,3,0:vw8); (1,0,0,0:vw8); (1,0,1,0:vw8); (1,0,2,0:vw8); (1,0,3,0:vw8); (2,0,0,0:vw8); (2,0,1,0:vw8); (2,0,2,0:vw8); (2,0,3,0:vw8) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v18, 5, v[vgprSerial]                // v18 = v[vgprSerial] * 32
s_mov_b32 s58, 0                                   // Init sgpr offset
buffer_load_dwordx4 v[120:123], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[124:127], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[128:131], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[132:135], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[136:139], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[140:143], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[144:147], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[148:151], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[152:155], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[156:159], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[160:163], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[164:167], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[168:171], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[172:175], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[176:179], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[180:183], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[184:187], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[188:191], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[200:203], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[204:207], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[208:211], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[212:215], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
s_add_u32 s58, s58, 8192                           // Inc sgpr offset
buffer_load_dwordx4 v[216:219], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:0 // load WS
buffer_load_dwordx4 v[220:223], v18, s[sgprSrdWS:sgprSrdWS+3], s58 offen offset:16 // load WS
v_accvgpr_read_b32 v[vgprValuC+24], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+25], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+26], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+27], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+28], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+29], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+30], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+31], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+44], acc18          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+45], acc22          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+46], acc26          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+47], acc30          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+48], acc3           // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+49], acc7           // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+50], acc11          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+51], acc15          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+52], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+53], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+54], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+55], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+56], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+57], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+58], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+59], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+60], acc48          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+61], acc52          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+62], acc56          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+63], acc60          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+64], acc33          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+65], acc37          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+66], acc41          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+67], acc45          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+68], acc49          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+69], acc53          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+70], acc57          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+71], acc61          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+76], acc50          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+77], acc54          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+78], acc58          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+79], acc62          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+80], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+81], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+82], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+83], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+84], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+85], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+86], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+87], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+88], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+89], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+90], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+91], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+92], acc80          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+93], acc84          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+94], acc88          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+95], acc92          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+96], acc65          // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+97], acc69          // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+98], acc73          // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+99], acc77          // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+100], acc81         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+101], acc85         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+102], acc89         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+103], acc93         // copy acc to vreg[79]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[80]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[81]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[82]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[83]
v_accvgpr_read_b32 v[vgprValuC+108], acc82         // copy acc to vreg[84]
v_accvgpr_read_b32 v[vgprValuC+109], acc86         // copy acc to vreg[85]
v_accvgpr_read_b32 v[vgprValuC+110], acc90         // copy acc to vreg[86]
v_accvgpr_read_b32 v[vgprValuC+111], acc94         // copy acc to vreg[87]
v_accvgpr_read_b32 v[vgprValuC+112], acc67         // copy acc to vreg[88]
v_accvgpr_read_b32 v[vgprValuC+113], acc71         // copy acc to vreg[89]
v_accvgpr_read_b32 v[vgprValuC+114], acc75         // copy acc to vreg[90]
v_accvgpr_read_b32 v[vgprValuC+115], acc79         // copy acc to vreg[91]
v_accvgpr_read_b32 v[vgprValuC+116], acc83         // copy acc to vreg[92]
v_accvgpr_read_b32 v[vgprValuC+117], acc87         // copy acc to vreg[93]
v_accvgpr_read_b32 v[vgprValuC+118], acc91         // copy acc to vreg[94]
v_accvgpr_read_b32 v[vgprValuC+119], acc95         // copy acc to vreg[95]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v14, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v15, 0x7fff0000                          // fp32 Nan
v_mov_b32 v16, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 0 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v120   // accum partials
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v121   // accum partials
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v122   // accum partials
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v123   // accum partials
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v124   // accum partials
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v125   // accum partials
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v126   // accum partials
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v127   // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 12 - 1 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v128   // accum partials
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v129   // accum partials
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v130   // accum partials
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v131   // accum partials
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v132   // accum partials
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v133   // accum partials
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v134   // accum partials
v_add_f32 v[vgprValuC+39], v[vgprValuC+39], v135   // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 12 - 2 + 0 - 1
v_add_f32 v[vgprValuC+40], v[vgprValuC+40], v136   // accum partials
v_add_f32 v[vgprValuC+41], v[vgprValuC+41], v137   // accum partials
v_add_f32 v[vgprValuC+42], v[vgprValuC+42], v138   // accum partials
v_add_f32 v[vgprValuC+43], v[vgprValuC+43], v139   // accum partials
v_add_f32 v[vgprValuC+44], v[vgprValuC+44], v140   // accum partials
v_add_f32 v[vgprValuC+45], v[vgprValuC+45], v141   // accum partials
v_add_f32 v[vgprValuC+46], v[vgprValuC+46], v142   // accum partials
v_add_f32 v[vgprValuC+47], v[vgprValuC+47], v143   // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 12 - 3 + 0 - 1
v_add_f32 v[vgprValuC+48], v[vgprValuC+48], v144   // accum partials
v_add_f32 v[vgprValuC+49], v[vgprValuC+49], v145   // accum partials
v_add_f32 v[vgprValuC+50], v[vgprValuC+50], v146   // accum partials
v_add_f32 v[vgprValuC+51], v[vgprValuC+51], v147   // accum partials
v_add_f32 v[vgprValuC+52], v[vgprValuC+52], v148   // accum partials
v_add_f32 v[vgprValuC+53], v[vgprValuC+53], v149   // accum partials
v_add_f32 v[vgprValuC+54], v[vgprValuC+54], v150   // accum partials
v_add_f32 v[vgprValuC+55], v[vgprValuC+55], v151   // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 12 - 4 + 0 - 1
v_add_f32 v[vgprValuC+56], v[vgprValuC+56], v152   // accum partials
v_add_f32 v[vgprValuC+57], v[vgprValuC+57], v153   // accum partials
v_add_f32 v[vgprValuC+58], v[vgprValuC+58], v154   // accum partials
v_add_f32 v[vgprValuC+59], v[vgprValuC+59], v155   // accum partials
v_add_f32 v[vgprValuC+60], v[vgprValuC+60], v156   // accum partials
v_add_f32 v[vgprValuC+61], v[vgprValuC+61], v157   // accum partials
v_add_f32 v[vgprValuC+62], v[vgprValuC+62], v158   // accum partials
v_add_f32 v[vgprValuC+63], v[vgprValuC+63], v159   // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 12 - 5 + 0 - 1
v_add_f32 v[vgprValuC+64], v[vgprValuC+64], v160   // accum partials
v_add_f32 v[vgprValuC+65], v[vgprValuC+65], v161   // accum partials
v_add_f32 v[vgprValuC+66], v[vgprValuC+66], v162   // accum partials
v_add_f32 v[vgprValuC+67], v[vgprValuC+67], v163   // accum partials
v_add_f32 v[vgprValuC+68], v[vgprValuC+68], v164   // accum partials
v_add_f32 v[vgprValuC+69], v[vgprValuC+69], v165   // accum partials
v_add_f32 v[vgprValuC+70], v[vgprValuC+70], v166   // accum partials
v_add_f32 v[vgprValuC+71], v[vgprValuC+71], v167   // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 12 - 6 + 0 - 1
v_add_f32 v[vgprValuC+72], v[vgprValuC+72], v168   // accum partials
v_add_f32 v[vgprValuC+73], v[vgprValuC+73], v169   // accum partials
v_add_f32 v[vgprValuC+74], v[vgprValuC+74], v170   // accum partials
v_add_f32 v[vgprValuC+75], v[vgprValuC+75], v171   // accum partials
v_add_f32 v[vgprValuC+76], v[vgprValuC+76], v172   // accum partials
v_add_f32 v[vgprValuC+77], v[vgprValuC+77], v173   // accum partials
v_add_f32 v[vgprValuC+78], v[vgprValuC+78], v174   // accum partials
v_add_f32 v[vgprValuC+79], v[vgprValuC+79], v175   // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 12 - 7 + 0 - 1
v_add_f32 v[vgprValuC+80], v[vgprValuC+80], v176   // accum partials
v_add_f32 v[vgprValuC+81], v[vgprValuC+81], v177   // accum partials
v_add_f32 v[vgprValuC+82], v[vgprValuC+82], v178   // accum partials
v_add_f32 v[vgprValuC+83], v[vgprValuC+83], v179   // accum partials
v_add_f32 v[vgprValuC+84], v[vgprValuC+84], v180   // accum partials
v_add_f32 v[vgprValuC+85], v[vgprValuC+85], v181   // accum partials
v_add_f32 v[vgprValuC+86], v[vgprValuC+86], v182   // accum partials
v_add_f32 v[vgprValuC+87], v[vgprValuC+87], v183   // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 12 - 8 + 0 - 1
v_add_f32 v[vgprValuC+88], v[vgprValuC+88], v184   // accum partials
v_add_f32 v[vgprValuC+89], v[vgprValuC+89], v185   // accum partials
v_add_f32 v[vgprValuC+90], v[vgprValuC+90], v186   // accum partials
v_add_f32 v[vgprValuC+91], v[vgprValuC+91], v187   // accum partials
v_add_f32 v[vgprValuC+92], v[vgprValuC+92], v188   // accum partials
v_add_f32 v[vgprValuC+93], v[vgprValuC+93], v189   // accum partials
v_add_f32 v[vgprValuC+94], v[vgprValuC+94], v190   // accum partials
v_add_f32 v[vgprValuC+95], v[vgprValuC+95], v191   // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 12 - 9 + 0 - 1
v_add_f32 v[vgprValuC+96], v[vgprValuC+96], v200   // accum partials
v_add_f32 v[vgprValuC+97], v[vgprValuC+97], v201   // accum partials
v_add_f32 v[vgprValuC+98], v[vgprValuC+98], v202   // accum partials
v_add_f32 v[vgprValuC+99], v[vgprValuC+99], v203   // accum partials
v_add_f32 v[vgprValuC+100], v[vgprValuC+100], v204 // accum partials
v_add_f32 v[vgprValuC+101], v[vgprValuC+101], v205 // accum partials
v_add_f32 v[vgprValuC+102], v[vgprValuC+102], v206 // accum partials
v_add_f32 v[vgprValuC+103], v[vgprValuC+103], v207 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 12 - 10 + 0 - 1
v_add_f32 v[vgprValuC+104], v[vgprValuC+104], v208 // accum partials
v_add_f32 v[vgprValuC+105], v[vgprValuC+105], v209 // accum partials
v_add_f32 v[vgprValuC+106], v[vgprValuC+106], v210 // accum partials
v_add_f32 v[vgprValuC+107], v[vgprValuC+107], v211 // accum partials
v_add_f32 v[vgprValuC+108], v[vgprValuC+108], v212 // accum partials
v_add_f32 v[vgprValuC+109], v[vgprValuC+109], v213 // accum partials
v_add_f32 v[vgprValuC+110], v[vgprValuC+110], v214 // accum partials
v_add_f32 v[vgprValuC+111], v[vgprValuC+111], v215 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 12 - 11 + 0 - 1
v_add_f32 v[vgprValuC+112], v[vgprValuC+112], v216 // accum partials
v_add_f32 v[vgprValuC+113], v[vgprValuC+113], v217 // accum partials
v_add_f32 v[vgprValuC+114], v[vgprValuC+114], v218 // accum partials
v_add_f32 v[vgprValuC+115], v[vgprValuC+115], v219 // accum partials
v_add_f32 v[vgprValuC+116], v[vgprValuC+116], v220 // accum partials
v_add_f32 v[vgprValuC+117], v[vgprValuC+117], v221 // accum partials
v_add_f32 v[vgprValuC+118], v[vgprValuC+118], v222 // accum partials
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v223 // accum partials
v_accvgpr_write_b32 acc0, v[vgprValuC+24]          // copy vreg[0] to acc
v_accvgpr_write_b32 acc4, v[vgprValuC+25]          // copy vreg[1] to acc
v_accvgpr_write_b32 acc8, v[vgprValuC+26]          // copy vreg[2] to acc
v_accvgpr_write_b32 acc12, v[vgprValuC+27]         // copy vreg[3] to acc
v_accvgpr_write_b32 acc16, v[vgprValuC+28]         // copy vreg[4] to acc
v_accvgpr_write_b32 acc20, v[vgprValuC+29]         // copy vreg[5] to acc
v_accvgpr_write_b32 acc24, v[vgprValuC+30]         // copy vreg[6] to acc
v_accvgpr_write_b32 acc28, v[vgprValuC+31]         // copy vreg[7] to acc
v_accvgpr_write_b32 acc1, v[vgprValuC+32]          // copy vreg[8] to acc
v_accvgpr_write_b32 acc5, v[vgprValuC+33]          // copy vreg[9] to acc
v_accvgpr_write_b32 acc9, v[vgprValuC+34]          // copy vreg[10] to acc
v_accvgpr_write_b32 acc13, v[vgprValuC+35]         // copy vreg[11] to acc
v_accvgpr_write_b32 acc17, v[vgprValuC+36]         // copy vreg[12] to acc
v_accvgpr_write_b32 acc21, v[vgprValuC+37]         // copy vreg[13] to acc
v_accvgpr_write_b32 acc25, v[vgprValuC+38]         // copy vreg[14] to acc
v_accvgpr_write_b32 acc29, v[vgprValuC+39]         // copy vreg[15] to acc
v_accvgpr_write_b32 acc2, v[vgprValuC+40]          // copy vreg[16] to acc
v_accvgpr_write_b32 acc6, v[vgprValuC+41]          // copy vreg[17] to acc
v_accvgpr_write_b32 acc10, v[vgprValuC+42]         // copy vreg[18] to acc
v_accvgpr_write_b32 acc14, v[vgprValuC+43]         // copy vreg[19] to acc
v_accvgpr_write_b32 acc18, v[vgprValuC+44]         // copy vreg[20] to acc
v_accvgpr_write_b32 acc22, v[vgprValuC+45]         // copy vreg[21] to acc
v_accvgpr_write_b32 acc26, v[vgprValuC+46]         // copy vreg[22] to acc
v_accvgpr_write_b32 acc30, v[vgprValuC+47]         // copy vreg[23] to acc
v_accvgpr_write_b32 acc3, v[vgprValuC+48]          // copy vreg[24] to acc
v_accvgpr_write_b32 acc7, v[vgprValuC+49]          // copy vreg[25] to acc
v_accvgpr_write_b32 acc11, v[vgprValuC+50]         // copy vreg[26] to acc
v_accvgpr_write_b32 acc15, v[vgprValuC+51]         // copy vreg[27] to acc
v_accvgpr_write_b32 acc19, v[vgprValuC+52]         // copy vreg[28] to acc
v_accvgpr_write_b32 acc23, v[vgprValuC+53]         // copy vreg[29] to acc
v_accvgpr_write_b32 acc27, v[vgprValuC+54]         // copy vreg[30] to acc
v_accvgpr_write_b32 acc31, v[vgprValuC+55]         // copy vreg[31] to acc
v_accvgpr_write_b32 acc32, v[vgprValuC+56]         // copy vreg[32] to acc
v_accvgpr_write_b32 acc36, v[vgprValuC+57]         // copy vreg[33] to acc
v_accvgpr_write_b32 acc40, v[vgprValuC+58]         // copy vreg[34] to acc
v_accvgpr_write_b32 acc44, v[vgprValuC+59]         // copy vreg[35] to acc
v_accvgpr_write_b32 acc48, v[vgprValuC+60]         // copy vreg[36] to acc
v_accvgpr_write_b32 acc52, v[vgprValuC+61]         // copy vreg[37] to acc
v_accvgpr_write_b32 acc56, v[vgprValuC+62]         // copy vreg[38] to acc
v_accvgpr_write_b32 acc60, v[vgprValuC+63]         // copy vreg[39] to acc
v_accvgpr_write_b32 acc33, v[vgprValuC+64]         // copy vreg[40] to acc
v_accvgpr_write_b32 acc37, v[vgprValuC+65]         // copy vreg[41] to acc
v_accvgpr_write_b32 acc41, v[vgprValuC+66]         // copy vreg[42] to acc
v_accvgpr_write_b32 acc45, v[vgprValuC+67]         // copy vreg[43] to acc
v_accvgpr_write_b32 acc49, v[vgprValuC+68]         // copy vreg[44] to acc
v_accvgpr_write_b32 acc53, v[vgprValuC+69]         // copy vreg[45] to acc
v_accvgpr_write_b32 acc57, v[vgprValuC+70]         // copy vreg[46] to acc
v_accvgpr_write_b32 acc61, v[vgprValuC+71]         // copy vreg[47] to acc
v_accvgpr_write_b32 acc34, v[vgprValuC+72]         // copy vreg[48] to acc
v_accvgpr_write_b32 acc38, v[vgprValuC+73]         // copy vreg[49] to acc
v_accvgpr_write_b32 acc42, v[vgprValuC+74]         // copy vreg[50] to acc
v_accvgpr_write_b32 acc46, v[vgprValuC+75]         // copy vreg[51] to acc
v_accvgpr_write_b32 acc50, v[vgprValuC+76]         // copy vreg[52] to acc
v_accvgpr_write_b32 acc54, v[vgprValuC+77]         // copy vreg[53] to acc
v_accvgpr_write_b32 acc58, v[vgprValuC+78]         // copy vreg[54] to acc
v_accvgpr_write_b32 acc62, v[vgprValuC+79]         // copy vreg[55] to acc
v_accvgpr_write_b32 acc35, v[vgprValuC+80]         // copy vreg[56] to acc
v_accvgpr_write_b32 acc39, v[vgprValuC+81]         // copy vreg[57] to acc
v_accvgpr_write_b32 acc43, v[vgprValuC+82]         // copy vreg[58] to acc
v_accvgpr_write_b32 acc47, v[vgprValuC+83]         // copy vreg[59] to acc
v_accvgpr_write_b32 acc51, v[vgprValuC+84]         // copy vreg[60] to acc
v_accvgpr_write_b32 acc55, v[vgprValuC+85]         // copy vreg[61] to acc
v_accvgpr_write_b32 acc59, v[vgprValuC+86]         // copy vreg[62] to acc
v_accvgpr_write_b32 acc63, v[vgprValuC+87]         // copy vreg[63] to acc
v_accvgpr_write_b32 acc64, v[vgprValuC+88]         // copy vreg[64] to acc
v_accvgpr_write_b32 acc68, v[vgprValuC+89]         // copy vreg[65] to acc
v_accvgpr_write_b32 acc72, v[vgprValuC+90]         // copy vreg[66] to acc
v_accvgpr_write_b32 acc76, v[vgprValuC+91]         // copy vreg[67] to acc
v_accvgpr_write_b32 acc80, v[vgprValuC+92]         // copy vreg[68] to acc
v_accvgpr_write_b32 acc84, v[vgprValuC+93]         // copy vreg[69] to acc
v_accvgpr_write_b32 acc88, v[vgprValuC+94]         // copy vreg[70] to acc
v_accvgpr_write_b32 acc92, v[vgprValuC+95]         // copy vreg[71] to acc
v_accvgpr_write_b32 acc65, v[vgprValuC+96]         // copy vreg[72] to acc
v_accvgpr_write_b32 acc69, v[vgprValuC+97]         // copy vreg[73] to acc
v_accvgpr_write_b32 acc73, v[vgprValuC+98]         // copy vreg[74] to acc
v_accvgpr_write_b32 acc77, v[vgprValuC+99]         // copy vreg[75] to acc
v_accvgpr_write_b32 acc81, v[vgprValuC+100]        // copy vreg[76] to acc
v_accvgpr_write_b32 acc85, v[vgprValuC+101]        // copy vreg[77] to acc
v_accvgpr_write_b32 acc89, v[vgprValuC+102]        // copy vreg[78] to acc
v_accvgpr_write_b32 acc93, v[vgprValuC+103]        // copy vreg[79] to acc
v_accvgpr_write_b32 acc66, v[vgprValuC+104]        // copy vreg[80] to acc
v_accvgpr_write_b32 acc70, v[vgprValuC+105]        // copy vreg[81] to acc
v_accvgpr_write_b32 acc74, v[vgprValuC+106]        // copy vreg[82] to acc
v_accvgpr_write_b32 acc78, v[vgprValuC+107]        // copy vreg[83] to acc
v_accvgpr_write_b32 acc82, v[vgprValuC+108]        // copy vreg[84] to acc
v_accvgpr_write_b32 acc86, v[vgprValuC+109]        // copy vreg[85] to acc
v_accvgpr_write_b32 acc90, v[vgprValuC+110]        // copy vreg[86] to acc
v_accvgpr_write_b32 acc94, v[vgprValuC+111]        // copy vreg[87] to acc
v_accvgpr_write_b32 acc67, v[vgprValuC+112]        // copy vreg[88] to acc
v_accvgpr_write_b32 acc71, v[vgprValuC+113]        // copy vreg[89] to acc
v_accvgpr_write_b32 acc75, v[vgprValuC+114]        // copy vreg[90] to acc
v_accvgpr_write_b32 acc79, v[vgprValuC+115]        // copy vreg[91] to acc
v_accvgpr_write_b32 acc83, v[vgprValuC+116]        // copy vreg[92] to acc
v_accvgpr_write_b32 acc87, v[vgprValuC+117]        // copy vreg[93] to acc
v_accvgpr_write_b32 acc91, v[vgprValuC+118]        // copy vreg[94] to acc
v_accvgpr_write_b32 acc95, v[vgprValuC+119]        // copy vreg[95] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s58, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s59, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s58, s58, s59                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_add_u32 s59, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s58                               // Check if next WG had an extra iteration
s_cselect_b32 s59, s59, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
s_add_u32 s9, s9, s59                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc1 label_NoBranch_F5UBNCGPE88CUZ4M     // Only branch on scc0
label_NoBranch_F5UBNCGPE88CUZ4M:

s_and_b32 s74, 127, s[sgprSizeI]                   // s74 = s[sgprSizeI] % 128
s_add_u32 s75, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s75                // wg0 >= nwg0-1 ?
s_cselect_b32 s74, s74, 0                          // set rMT0
s_cmpk_gt_u32 s74, 0                               // rMT0 > 0
s_cbranch_scc0 label_NoBranch_47BMNH95JUTHU93Y     // Only branch on scc1
// jump if edges required
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_GW_B0_E1_M_1, 4               // target branch offset
s_add_u32 s74, s74, s76                            // add target branch offset
s_addc_u32 s75, s75, 0                             // add high and carry
s_setpc_b64 s[74:75]                               // branch to label_GW_B0_E1_M_1
label_NoBranch_47BMNH95JUTHU93Y:
s_mov_b32 s77, 0                                   // STATIC_DIV: divisor=192
s_mul_i32 s76, 682, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[76:77], s[76:77], 16                  // left shift 16 bits
s_mul_i32 s75, s[sgprSizeJ], 43691                 // tmp0 = dividend * magic lo
s_add_u32 s76, s75, s76                            // add lo
s_addc_u32 s77, s77, 0                             // add hi
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s75, s76                                 // quotient
s_mul_i32 s76, s75, 192                            // quotient*divisor
s_sub_u32 s74, s[sgprSizeJ], s76                   // rReg = dividend - quotient*divisor
s_add_u32 s75, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s75                // wg1 >= nwg1-1
s_cselect_b32 s74, s74, 0                          // set rMT1
s_cmpk_gt_u32 s74, 0                               // rMT1 > 0
s_cbranch_scc0 label_NoBranch_NJCHQCCFCCMCRV12     // Only branch on scc1
// jump if edges required
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_GW_B0_E1_N_1, 4               // target branch offset
s_add_u32 s74, s74, s76                            // add target branch offset
s_addc_u32 s75, s75, 0                             // add high and carry
s_setpc_b64 s[74:75]                               // branch to label_GW_B0_E1_N_1
label_NoBranch_NJCHQCCFCCMCRV12:
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=8 */
s_cmpk_eq_u32 s[sgprActivationType], 0             // activationType == 0
s_cbranch_scc1 label_Activation_None_1             // Branch if true
label_Activation_None_1:
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8); (0,0,1,0:vw8); (0,0,2,0:vw8); (0,0,3,0:vw8); (1,0,0,0:vw8); (1,0,1,0:vw8); (1,0,2,0:vw8); (1,0,3,0:vw8) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v19, v0, s8
v_lshlrev_b32 v19, 0x2, v19                        // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b128 v[88:91], v19 offset:0                // load Bias
ds_read_b128 v[92:95], v19 offset:16               // load Bias
ds_read_b128 v[96:99], v19 offset:1024             // load scaleAlpha
ds_read_b128 v[100:103], v19 offset:1040           // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_lshl_u32 v17, v3, v0, 0x1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+24], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+25], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+26], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+27], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+28], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+29], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+30], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+31], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+44], acc18          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+45], acc22          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+46], acc26          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+47], acc30          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+48], acc3           // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+49], acc7           // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+50], acc11          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+51], acc15          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+52], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+53], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+54], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+55], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+56], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+57], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+58], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+59], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+60], acc48          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+61], acc52          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+62], acc56          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+63], acc60          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+64], acc33          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+65], acc37          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+66], acc41          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+67], acc45          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+68], acc49          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+69], acc53          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+70], acc57          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+71], acc61          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+76], acc50          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+77], acc54          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+78], acc58          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+79], acc62          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+80], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+81], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+82], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+83], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+84], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+85], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+86], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+87], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)

// label_jzhou:
/******************************************/
/* Macro: Increment SRD to Next Row(s)   */
/******************************************/
/* Parameters:
 *   numRows  - number of rows to skip (1, 16, 61, 1024, etc.)
 *   tmpSgpr  - temporary scalar register to use (e.g., s8)
 *   srdBase  - base of SRD register (e.g., sgprSrdD)
 *   stride   - stride register (e.g., sgprStrideD1J)
 */

.macro incToNextRow numRows, tmpSgpr=s8, srdBase=sgprSrdD, stride=sgprStrideD1J
  .if \numRows == 1
    // Jump 1 row: stride * 2 (BPE)
    s_lshl_b32 \tmpSgpr, s[\stride], 1
  .elseif \numRows == 16
    // Jump 16 rows: stride * 32 (16 * 2 BPE)
    s_lshl_b32 \tmpSgpr, s[\stride], 5
  .elseif \numRows == 61
    // Jump 61 rows: stride * 122 (61 * 2 BPE)
    s_mul_i32 \tmpSgpr, s[\stride], 122
  .elseif \numRows == 1024
    // Jump 1024 rows: stride * 2048 (1024 * 2 BPE)
    s_mul_i32 \tmpSgpr, s[\stride], 2048
  .else
    // General case: stride * (numRows * 2)
    .set _offset, \numRows * 2
    s_mul_i32 \tmpSgpr, s[\stride], _offset
  .endif
  // Add offset to SRD base address (64-bit)
//  s_add_u32 s[\srdBase+0], s[\srdBase+0], \tmpSgpr
 // s_addc_u32 s[\srdBase+1], s[\srdBase+1], 0
.endm

/* apply mask, calc new C and issue writes */
v_mov_b32 v14, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v15, 0x7fff0000                          // fp32 Nan
v_mov_b32 v16, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 4 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], v[96:97], v[vgprValuC+24:vgprValuC+24+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], v[98:99], v[vgprValuC+26:vgprValuC+26+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], v[100:101], v[vgprValuC+28:vgprValuC+28+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], v[102:103], v[vgprValuC+30:vgprValuC+30+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+24:vgprValuC+24+1], v[88:89], v[vgprValuC+24:vgprValuC+24+1] // C += bias
v_pk_add_f32 v[vgprValuC+26:vgprValuC+26+1], v[90:91], v[vgprValuC+26:vgprValuC+26+1] // C += bias
v_pk_add_f32 v[vgprValuC+28:vgprValuC+28+1], v[92:93], v[vgprValuC+28:vgprValuC+28+1] // C += bias
v_pk_add_f32 v[vgprValuC+30:vgprValuC+30+1], v[94:95], v[vgprValuC+30:vgprValuC+30+1] // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
buffer_store_dwordx4 v[24:27], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], v[96:97], v[vgprValuC+32:vgprValuC+32+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], v[98:99], v[vgprValuC+34:vgprValuC+34+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], v[100:101], v[vgprValuC+36:vgprValuC+36+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], v[102:103], v[vgprValuC+38:vgprValuC+38+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+32:vgprValuC+32+1], v[88:89], v[vgprValuC+32:vgprValuC+32+1] // C += bias
v_pk_add_f32 v[vgprValuC+34:vgprValuC+34+1], v[90:91], v[vgprValuC+34:vgprValuC+34+1] // C += bias
v_pk_add_f32 v[vgprValuC+36:vgprValuC+36+1], v[92:93], v[vgprValuC+36:vgprValuC+36+1] // C += bias
v_pk_add_f32 v[vgprValuC+38:vgprValuC+38+1], v[94:95], v[vgprValuC+38:vgprValuC+38+1] // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v34, v[vgprValuC+36], v[vgprValuC+37] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v35, v[vgprValuC+38], v[vgprValuC+39] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], v[96:97], v[vgprValuC+40:vgprValuC+40+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], v[98:99], v[vgprValuC+42:vgprValuC+42+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], v[100:101], v[vgprValuC+44:vgprValuC+44+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], v[102:103], v[vgprValuC+46:vgprValuC+46+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+40:vgprValuC+40+1], v[88:89], v[vgprValuC+40:vgprValuC+40+1] // C += bias
v_pk_add_f32 v[vgprValuC+42:vgprValuC+42+1], v[90:91], v[vgprValuC+42:vgprValuC+42+1] // C += bias
v_pk_add_f32 v[vgprValuC+44:vgprValuC+44+1], v[92:93], v[vgprValuC+44:vgprValuC+44+1] // C += bias
v_pk_add_f32 v[vgprValuC+46:vgprValuC+46+1], v[94:95], v[vgprValuC+46:vgprValuC+46+1] // C += bias
v_cvt_pk_bf16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v42, v[vgprValuC+44], v[vgprValuC+45] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v43, v[vgprValuC+46], v[vgprValuC+47] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], v[96:97], v[vgprValuC+48:vgprValuC+48+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], v[98:99], v[vgprValuC+50:vgprValuC+50+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], v[100:101], v[vgprValuC+52:vgprValuC+52+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], v[102:103], v[vgprValuC+54:vgprValuC+54+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+48:vgprValuC+48+1], v[88:89], v[vgprValuC+48:vgprValuC+48+1] // C += bias
v_pk_add_f32 v[vgprValuC+50:vgprValuC+50+1], v[90:91], v[vgprValuC+50:vgprValuC+50+1] // C += bias
v_pk_add_f32 v[vgprValuC+52:vgprValuC+52+1], v[92:93], v[vgprValuC+52:vgprValuC+52+1] // C += bias
v_pk_add_f32 v[vgprValuC+54:vgprValuC+54+1], v[94:95], v[vgprValuC+54:vgprValuC+54+1] // C += bias
v_cvt_pk_bf16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v50, v[vgprValuC+52], v[vgprValuC+53] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v51, v[vgprValuC+54], v[vgprValuC+55] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], v[96:97], v[vgprValuC+56:vgprValuC+56+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], v[98:99], v[vgprValuC+58:vgprValuC+58+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], v[100:101], v[vgprValuC+60:vgprValuC+60+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], v[102:103], v[vgprValuC+62:vgprValuC+62+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+56:vgprValuC+56+1], v[88:89], v[vgprValuC+56:vgprValuC+56+1] // C += bias
v_pk_add_f32 v[vgprValuC+58:vgprValuC+58+1], v[90:91], v[vgprValuC+58:vgprValuC+58+1] // C += bias
v_pk_add_f32 v[vgprValuC+60:vgprValuC+60+1], v[92:93], v[vgprValuC+60:vgprValuC+60+1] // C += bias
v_pk_add_f32 v[vgprValuC+62:vgprValuC+62+1], v[94:95], v[vgprValuC+62:vgprValuC+62+1] // C += bias
v_cvt_pk_bf16_f32 v56, v[vgprValuC+56], v[vgprValuC+57] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v57, v[vgprValuC+58], v[vgprValuC+59] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v58, v[vgprValuC+60], v[vgprValuC+61] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v59, v[vgprValuC+62], v[vgprValuC+63] // convert C to bf16 and Pack with neighbor
incToNextRow 976
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], v[96:97], v[vgprValuC+64:vgprValuC+64+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], v[98:99], v[vgprValuC+66:vgprValuC+66+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], v[100:101], v[vgprValuC+68:vgprValuC+68+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], v[102:103], v[vgprValuC+70:vgprValuC+70+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+64:vgprValuC+64+1], v[88:89], v[vgprValuC+64:vgprValuC+64+1] // C += bias
v_pk_add_f32 v[vgprValuC+66:vgprValuC+66+1], v[90:91], v[vgprValuC+66:vgprValuC+66+1] // C += bias
v_pk_add_f32 v[vgprValuC+68:vgprValuC+68+1], v[92:93], v[vgprValuC+68:vgprValuC+68+1] // C += bias
v_pk_add_f32 v[vgprValuC+70:vgprValuC+70+1], v[94:95], v[vgprValuC+70:vgprValuC+70+1] // C += bias
v_cvt_pk_bf16_f32 v64, v[vgprValuC+64], v[vgprValuC+65] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v65, v[vgprValuC+66], v[vgprValuC+67] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v66, v[vgprValuC+68], v[vgprValuC+69] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v67, v[vgprValuC+70], v[vgprValuC+71] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], v[96:97], v[vgprValuC+72:vgprValuC+72+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], v[98:99], v[vgprValuC+74:vgprValuC+74+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], v[100:101], v[vgprValuC+76:vgprValuC+76+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], v[102:103], v[vgprValuC+78:vgprValuC+78+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+72:vgprValuC+72+1], v[88:89], v[vgprValuC+72:vgprValuC+72+1] // C += bias
v_pk_add_f32 v[vgprValuC+74:vgprValuC+74+1], v[90:91], v[vgprValuC+74:vgprValuC+74+1] // C += bias
v_pk_add_f32 v[vgprValuC+76:vgprValuC+76+1], v[92:93], v[vgprValuC+76:vgprValuC+76+1] // C += bias
v_pk_add_f32 v[vgprValuC+78:vgprValuC+78+1], v[94:95], v[vgprValuC+78:vgprValuC+78+1] // C += bias
v_cvt_pk_bf16_f32 v72, v[vgprValuC+72], v[vgprValuC+73] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v73, v[vgprValuC+74], v[vgprValuC+75] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v74, v[vgprValuC+76], v[vgprValuC+77] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v75, v[vgprValuC+78], v[vgprValuC+79] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[72:75], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], v[96:97], v[vgprValuC+80:vgprValuC+80+1] // *= ScaleAlphaVecVMulPK(96)(0)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], v[98:99], v[vgprValuC+82:vgprValuC+82+1] // *= ScaleAlphaVecVMulPK(96)(2)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], v[100:101], v[vgprValuC+84:vgprValuC+84+1] // *= ScaleAlphaVecVMulPK(96)(4)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], v[102:103], v[vgprValuC+86:vgprValuC+86+1] // *= ScaleAlphaVecVMulPK(96)(6)
v_pk_add_f32 v[vgprValuC+80:vgprValuC+80+1], v[88:89], v[vgprValuC+80:vgprValuC+80+1] // C += bias
v_pk_add_f32 v[vgprValuC+82:vgprValuC+82+1], v[90:91], v[vgprValuC+82:vgprValuC+82+1] // C += bias
v_pk_add_f32 v[vgprValuC+84:vgprValuC+84+1], v[92:93], v[vgprValuC+84:vgprValuC+84+1] // C += bias
v_pk_add_f32 v[vgprValuC+86:vgprValuC+86+1], v[94:95], v[vgprValuC+86:vgprValuC+86+1] // C += bias
v_cvt_pk_bf16_f32 v80, v[vgprValuC+80], v[vgprValuC+81] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v81, v[vgprValuC+82], v[vgprValuC+83] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v82, v[vgprValuC+84], v[vgprValuC+85] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v83, v[vgprValuC+86], v[vgprValuC+87] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[80:83], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw8); (2,0,1,0:vw8); (2,0,2,0:vw8); (2,0,3,0:vw8) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
ds_read_b128 v[56:59], v19 offset:0                // load Bias
ds_read_b128 v[60:63], v19 offset:16               // load Bias
ds_read_b128 v[64:67], v19 offset:1024             // load scaleAlpha
ds_read_b128 v[68:71], v19 offset:1040             // load scaleAlpha
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_accvgpr_read_b32 v[vgprValuC+24], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+25], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+26], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+27], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+28], acc80          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+29], acc84          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+30], acc88          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+31], acc92          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+32], acc65          // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+33], acc69          // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+34], acc73          // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+35], acc77          // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+36], acc81          // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+37], acc85          // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+38], acc89          // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+39], acc93          // copy acc to vreg[79]
v_accvgpr_read_b32 v[vgprValuC+40], acc66          // copy acc to vreg[80]
v_accvgpr_read_b32 v[vgprValuC+41], acc70          // copy acc to vreg[81]
v_accvgpr_read_b32 v[vgprValuC+42], acc74          // copy acc to vreg[82]
v_accvgpr_read_b32 v[vgprValuC+43], acc78          // copy acc to vreg[83]
v_accvgpr_read_b32 v[vgprValuC+44], acc82          // copy acc to vreg[84]
v_accvgpr_read_b32 v[vgprValuC+45], acc86          // copy acc to vreg[85]
v_accvgpr_read_b32 v[vgprValuC+46], acc90          // copy acc to vreg[86]
v_accvgpr_read_b32 v[vgprValuC+47], acc94          // copy acc to vreg[87]
v_accvgpr_read_b32 v[vgprValuC+48], acc67          // copy acc to vreg[88]
v_accvgpr_read_b32 v[vgprValuC+49], acc71          // copy acc to vreg[89]
v_accvgpr_read_b32 v[vgprValuC+50], acc75          // copy acc to vreg[90]
v_accvgpr_read_b32 v[vgprValuC+51], acc79          // copy acc to vreg[91]
v_accvgpr_read_b32 v[vgprValuC+52], acc83          // copy acc to vreg[92]
v_accvgpr_read_b32 v[vgprValuC+53], acc87          // copy acc to vreg[93]
v_accvgpr_read_b32 v[vgprValuC+54], acc91          // copy acc to vreg[94]
v_accvgpr_read_b32 v[vgprValuC+55], acc95          // copy acc to vreg[95]

/* rC *= alpha batchElements=[(2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v14, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v15, 0x7fff0000                          // fp32 Nan
v_mov_b32 v16, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 4 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], v[64:65], v[vgprValuC+24:vgprValuC+24+1] // *= ScaleAlphaVecVMulPK(64)(0)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], v[66:67], v[vgprValuC+26:vgprValuC+26+1] // *= ScaleAlphaVecVMulPK(64)(2)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], v[68:69], v[vgprValuC+28:vgprValuC+28+1] // *= ScaleAlphaVecVMulPK(64)(4)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], v[70:71], v[vgprValuC+30:vgprValuC+30+1] // *= ScaleAlphaVecVMulPK(64)(6)
v_pk_add_f32 v[vgprValuC+24:vgprValuC+24+1], v[56:57], v[vgprValuC+24:vgprValuC+24+1] // C += bias
v_pk_add_f32 v[vgprValuC+26:vgprValuC+26+1], v[58:59], v[vgprValuC+26:vgprValuC+26+1] // C += bias
v_pk_add_f32 v[vgprValuC+28:vgprValuC+28+1], v[60:61], v[vgprValuC+28:vgprValuC+28+1] // C += bias
v_pk_add_f32 v[vgprValuC+30:vgprValuC+30+1], v[62:63], v[vgprValuC+30:vgprValuC+30+1] // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
incToNextRow 976
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], v[64:65], v[vgprValuC+32:vgprValuC+32+1] // *= ScaleAlphaVecVMulPK(64)(0)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], v[66:67], v[vgprValuC+34:vgprValuC+34+1] // *= ScaleAlphaVecVMulPK(64)(2)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], v[68:69], v[vgprValuC+36:vgprValuC+36+1] // *= ScaleAlphaVecVMulPK(64)(4)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], v[70:71], v[vgprValuC+38:vgprValuC+38+1] // *= ScaleAlphaVecVMulPK(64)(6)
v_pk_add_f32 v[vgprValuC+32:vgprValuC+32+1], v[56:57], v[vgprValuC+32:vgprValuC+32+1] // C += bias
v_pk_add_f32 v[vgprValuC+34:vgprValuC+34+1], v[58:59], v[vgprValuC+34:vgprValuC+34+1] // C += bias
v_pk_add_f32 v[vgprValuC+36:vgprValuC+36+1], v[60:61], v[vgprValuC+36:vgprValuC+36+1] // C += bias
v_pk_add_f32 v[vgprValuC+38:vgprValuC+38+1], v[62:63], v[vgprValuC+38:vgprValuC+38+1] // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v34, v[vgprValuC+36], v[vgprValuC+37] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v35, v[vgprValuC+38], v[vgprValuC+39] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], v[64:65], v[vgprValuC+40:vgprValuC+40+1] // *= ScaleAlphaVecVMulPK(64)(0)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], v[66:67], v[vgprValuC+42:vgprValuC+42+1] // *= ScaleAlphaVecVMulPK(64)(2)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], v[68:69], v[vgprValuC+44:vgprValuC+44+1] // *= ScaleAlphaVecVMulPK(64)(4)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], v[70:71], v[vgprValuC+46:vgprValuC+46+1] // *= ScaleAlphaVecVMulPK(64)(6)
v_pk_add_f32 v[vgprValuC+40:vgprValuC+40+1], v[56:57], v[vgprValuC+40:vgprValuC+40+1] // C += bias
v_pk_add_f32 v[vgprValuC+42:vgprValuC+42+1], v[58:59], v[vgprValuC+42:vgprValuC+42+1] // C += bias
v_pk_add_f32 v[vgprValuC+44:vgprValuC+44+1], v[60:61], v[vgprValuC+44:vgprValuC+44+1] // C += bias
v_pk_add_f32 v[vgprValuC+46:vgprValuC+46+1], v[62:63], v[vgprValuC+46:vgprValuC+46+1] // C += bias
v_cvt_pk_bf16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v42, v[vgprValuC+44], v[vgprValuC+45] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v43, v[vgprValuC+46], v[vgprValuC+47] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], v[64:65], v[vgprValuC+48:vgprValuC+48+1] // *= ScaleAlphaVecVMulPK(64)(0)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], v[66:67], v[vgprValuC+50:vgprValuC+50+1] // *= ScaleAlphaVecVMulPK(64)(2)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], v[68:69], v[vgprValuC+52:vgprValuC+52+1] // *= ScaleAlphaVecVMulPK(64)(4)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], v[70:71], v[vgprValuC+54:vgprValuC+54+1] // *= ScaleAlphaVecVMulPK(64)(6)
v_pk_add_f32 v[vgprValuC+48:vgprValuC+48+1], v[56:57], v[vgprValuC+48:vgprValuC+48+1] // C += bias
v_pk_add_f32 v[vgprValuC+50:vgprValuC+50+1], v[58:59], v[vgprValuC+50:vgprValuC+50+1] // C += bias
v_pk_add_f32 v[vgprValuC+52:vgprValuC+52+1], v[60:61], v[vgprValuC+52:vgprValuC+52+1] // C += bias
v_pk_add_f32 v[vgprValuC+54:vgprValuC+54+1], v[62:63], v[vgprValuC+54:vgprValuC+54+1] // C += bias
v_cvt_pk_bf16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v50, v[vgprValuC+52], v[vgprValuC+53] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v51, v[vgprValuC+54], v[vgprValuC+55] // convert C to bf16 and Pack with neighbor
incToNextRow 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst


// jump to end
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_GW_End_1, 4                   // target branch offset
s_add_u32 s74, s74, s76                            // add target branch offset
s_addc_u32 s75, s75, 0                             // add high and carry
s_setpc_b64 s[74:75]                               // branch to label_GW_End_1
label_GW_B0_E1_N_1:
label_GW_B0_E1_M_1:
label_GW_End_1:
label_SK_CloseLoop:
s_cmp_ge_u32 s[sgprStreamKIter], s[sgprStreamKIterEnd] // Check if done all StreamK iterations
s_cbranch_scc1 label_NoBranch_T4I764MR3KTNP3FE     // Only branch on scc0
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s76, s76                                 // abs offset
s_sub_u32 s74, s74, s76                            // sub target branch offset
s_subb_u32 s75, s75, 0                             // sub high and carry
s_setpc_b64 s[74:75]                               // branch to label_PersistentLoopStart
label_NoBranch_T4I764MR3KTNP3FE:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
