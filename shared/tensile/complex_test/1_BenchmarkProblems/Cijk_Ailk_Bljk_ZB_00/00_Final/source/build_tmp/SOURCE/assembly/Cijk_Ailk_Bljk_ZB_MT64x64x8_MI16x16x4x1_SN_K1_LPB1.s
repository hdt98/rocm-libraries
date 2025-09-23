
/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureDefault
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1
.globl Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1
.p2align 8
.type Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_user_sgpr_kernarg_preload_length 0
  .amdhsa_user_sgpr_count 2
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 320 // vgprs
  .amdhsa_next_free_sgpr 71 // sgprs
  .amdhsa_group_segment_fixed_size 49280 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 2 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalLoadVectorWidthA=1, GlobalLoadVectorWidthB=1 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.kernels:
  - .name: Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1
    .symbol: 'Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Tensor2dSizeA
        .size:            8
        .offset:          0
        .value_kind:      by_value
        .value_type:      u64
      - .name:            Tensor2dSizeB
        .size:            8
        .offset:          8
        .value_kind:      by_value
        .value_type:      u64
      - .name:            AddressD
        .size:            8
        .offset:          16
        .value_kind:      by_value
        .value_type:      u64
      - .name:            AddressC
        .size:            8
        .offset:          24
        .value_kind:      by_value
        .value_type:      u64
      - .name:            AddressA
        .size:            8
        .offset:          32
        .value_kind:      by_value
        .value_type:      u64
      - .name:            AddressB
        .size:            8
        .offset:          40
        .value_kind:      by_value
        .value_type:      u64
      - .name:            Alpha
        .size:            16
        .offset:          48
        .value_kind:      by_value
        .value_type:      u128
      - .name:            Beta
        .size:            16
        .offset:          64
        .value_kind:      by_value
        .value_type:      u128
      - .name:            StridesD
        .size:            8
        .offset:          80
        .value_kind:      by_value
        .value_type:      u64
      - .name:            StridesC
        .size:            8
        .offset:          88
        .value_kind:      by_value
        .value_type:      u64
      - .name:            StridesA
        .size:            8
        .offset:          96
        .value_kind:      by_value
        .value_type:      u64
      - .name:            StridesB
        .size:            8
        .offset:          104
        .value_kind:      by_value
        .value_type:      u64
      - .name:            SizesFree
        .size:            12
        .offset:          112
        .value_kind:      by_value
        .value_type:      u96
      - .name:            SizesSum
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          148
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   49280
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 71
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
.endm

.macro _v_dot2acc_f32_f16 dst, src0, src1
v_dot2c_f32_f16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmpx_ne_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmpx_ne_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmpx_ne_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmpx_ne_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmpx_ne_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmpx_ne_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst, \src0, \src1 
.endm
.macro _v_mac_f32 c:req, a:req, b:req
    v_fmac_f32 \c, \a, \b
.endmacro

/* scale global load macros */
.macro _s_load_b32 dst base offset
    s_load_dword \dst \base \offset
.endm

.macro _s_load_b64 dst base offset
    s_load_dwordx2 \dst \base \offset
.endm

.macro _s_load_b128 dst base offset
    s_load_dwordx4 \dst \base \offset
.endm

.macro _s_load_b256 dst base offset
    s_load_dwordx8 \dst \base \offset
.endm

.macro _s_load_b512 dst base offset
    s_load_dwordx16 \dst \base \offset
.endm


/* ds operation macros */
.macro _ds_load_u8 dst src offset
    ds_read_u8 \dst \src \offset
.endm

.macro _ds_load_u8_d16_hi dst src offset
    ds_read_u8_d16_hi \dst \src \offset
.endm

.macro _ds_load_u16 dst src offset
    ds_read_u16 \dst \src \offset
.endm

.macro _ds_load_u16_d16_hi dst src offset
    ds_read_u16_d16_hi \dst \src \offset
.endm

.macro _ds_load_b32 dst src offset
    ds_read_b32 \dst \src \offset
.endm

.macro _ds_load_b64 dst src offset
    ds_read_b64 \dst \src \offset
.endm

.macro _ds_load_b128 dst src offset
    ds_read_b128 \dst \src \offset
.endm

.macro _ds_store_b8 dst src offset
    ds_write_b8 \dst \src \offset
.endm

.macro _ds_store_b8_d16_hi dst src offset
    ds_write_b8_d16_hi \dst \src \offset
.endm

.macro _ds_store_b16 dst src offset
    ds_write_b16 \dst \src \offset
.endm

.macro _ds_store_b16_d16_hi dst src offset
    ds_write_b16_d16_hi \dst \src \offset
.endm

.macro _ds_store_b32 dst src offset
    ds_write_b32 \dst \src \offset
.endm

.macro _ds_store_b64 dst src offset
    ds_write_b64 \dst \src \offset
.endm

.macro _ds_store_b128 dst src offset
    ds_write_b128 \dst \src \offset
.endm

.macro _ds_load2_b32 dst src offset1 offset2
    ds_read2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_load2_b64 dst src offset1 offset2
    ds_read2_b64 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b32 dst src offset1 offset2
    ds_write2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b64 dst src offset1 offset2
    ds_write2_b64 \dst \src \offset1 \offset2
.endm


/* buffer memory operation macros */
.macro _buffer_load_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dword \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx2 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b96 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx3 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b128 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx4 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_short_d16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_short_d16_hi \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ubyte_d16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ubyte_d16_hi \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_u16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ushort \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b32_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dword \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b64_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx2 \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b128_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx4 \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_u16_dtl voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ushort \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b32 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dword \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b64 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx2 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b96 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx3 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b128 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx4 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_short \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_short_d16_hi \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_byte \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_byte_d16_hi \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap_x2 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm


/* buffer memory operation macros */
.macro _global_load_b32 dst base src ioffset md0 md1 md2
    global_load_dword \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b64 dst base src ioffset md0 md1 md2
    global_load_dwordx2 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b96 dst base src ioffset md0 md1 md2
    global_load_dwordx3 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_b128 dst base src ioffset md0 md1 md2
    global_load_dwordx4 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_b16 dst base src ioffset md0 md1 md2
    global_load_short_d16 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_hi_b16 dst base src ioffset md0 md1 md2
    global_load_short_d16_hi \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_u8 dst base src ioffset md0 md1 md2
    global_load_ubyte_d16 \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_d16_hi_u8 dst base src ioffset md0 md1 md2
    global_load_ubyte_d16_hi \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_load_u16 dst base src ioffset md0 md1 md2
    global_load_ushort \dst \base \src \ioffset \md0 \md1 \md2
.endm

.macro _global_store_b32 base src src2 md0 md1 md2
    global_store_dword \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b64 base src src2 md0 md1 md2
    global_store_dwordx2 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b96 base src src2 md0 md1 md2
    global_store_dwordx3 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_b128 base src src2 md0 md1 md2
    global_store_dwordx4 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_b16 base src src2 md0 md1 md2
    global_store_short \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_hi_b16 base src src2 md0 md1 md2
    global_store_short_d16_hi \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_u8 base src src2 md0 md1 md2
    global_store_ubyte_d16 \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_d16_hi_u8 base src src2 md0 md1 md2
    global_store_ubyte_d16_hi \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_store_u16 base src src2 md0 md1 md2
    global_store_ushort \base \src \src2 \md0 \md1 \md2
.endm

.macro _global_atomic_cmpswap_b32 tmp base data src ioffset md
    global_atomic_cmpswap \tmp \base \data \src \ioffset \md
.endm

.macro _global_atomic_cmpswap_b64 tmp base data src ioffset md
    global_atomic_cmpswap_x2 \tmp \base \data \src \ioffset \md
.endm


/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 0
.set vgprG2LA, 20
.set vgprValuB_X0_I0, 8
.set vgprG2LB, 28
.set vgprLocalWriteAddrA, 16
.set vgprLocalWriteAddrB, 17
.set vgprGlobalReadOffsetA, 18
.set vgprGlobalReadOffsetB, 19
.set vgprLocalReadAddrA, 36
.set vgprLocalReadAddrB, 37
.set vgprSerial, 38
/* Num VGPR=256 */
/* Num AccVGPR=64 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0 // (2)
.set sgprWorkGroup0, 2 // (1)
.set sgprWorkGroup1, 3 // (1)
.set sgprWorkGroup2, 4 // (1)
.set sgprLoopCounterL, 5 // (1)
.set sgprOrigLoopCounter, 6 // (1)
.set sgprSrdA, 8 // (4)
.set sgprSrdB, 12 // (4)
.set sgprSrdD, 16 // (4)
.set sgprSrdC, 20 // (4)
.set sgprTensor2dSizeA, 24 // (2)
.set sgprTensor2dSizeB, 26 // (2)
.set sgprAddressD, 28 // (2)
.set sgprAddressC, 30 // (2)
.set sgprAddressA, 32 // (2)
.set sgprAddressB, 34 // (2)
.set sgprAlpha, 36 // (4)
.set sgprBeta, 40 // (4)
.set sgprStridesD, 44 // (2)
.set sgprStridesC, 46 // (2)
.set sgprStridesA, 48 // (2)
.set sgprStridesB, 50 // (2)
.set sgprSizesFree, 52 // (3)
.set sgprSizesSum, 55 // (1)
.set sgprOrigStaggerUIter, 56 // (1)
.set sgprNumWorkGroups0, 57 // (1)
.set sgprNumWorkGroups1, 58 // (1)
.set sgprNumFullBlocks, 59 // (1)
.set sgprWgmRemainder1, 60 // (1)
.set sgprMagicNumberWgmRemainder1, 61 // (1)
.set sgprShadowLimitA, 0 // (2)
.set sgprShadowLimitB, 28 // (2)
.set sgprStaggerUIter, 7 // (1)
.set sgprWrapUA, 30 // (2)
.set sgprWrapUB, 32 // (2)
.set sgprGlobalReadIncsA, 34 // (1)
.set sgprGlobalReadIncsB, 35 // (1)
.set sgprScalarGlobalReadOffsetA, 64 // (1)
.set sgprScalarGlobalReadOffsetB, 65 // (1)
/* max SGPR=71 */

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
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 64
.set MT1, 64
.set DepthU, 8
.set GSU, 1
.set BpeA, 16
.set BpeALog2, 4
.set BpeB, 16
.set BpeBLog2, 4
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 1
.set SrdShiftLeftB, 1
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimitA, 0xffffffff
.set BufferLimitB, 0xffffffff
.set BufferOOB, 0xfffff000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
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
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          // 
v_rcp_f32 v[\vQuotient], v[\vQuotient]             // 
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] // 
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         // 
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] // 
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]  // 
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]      // 
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder] // 
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder] // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] // 
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder] // 
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder] // 
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient] // 
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]    // 
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]          // 
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc               // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc // 
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1] // 
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]                // 
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder] // final result
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

Cijk_Ailk_Bljk_ZB_MT64x64x8_MI16x16x4x1_SN_K1_LPB1_preloaded: // Kernel start when preloading

/* Load Kernel Args */
_s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 // 
_s_load_b512 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
_s_load_b128 s[56:59], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // 
_s_load_b64 s[60:61], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // 
s_mov_b32 m0, 0xc080                               // LDS clamp at 49280 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

/*lr0I*/
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x6, v1                          // 5. K offset: lrKOffset = kIdx * mStride(64)
_v_add_u32 v0, v1, v0                              // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 1, v2                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v1, 0x4, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16)
_v_add_u32 v0, v1, v0                              // 8. final local read offset: flrOffset = lrOffset + WOffset
/*lr1J*/
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
s_mov_b32 s7, 0x41                                 // 5. K offset: lrKOffset = kIdx * mStride(65)
v_mul_lo_u32 v2, s7, v2                            // 5. K offset: lrKOffset = kIdx * mStride(65)
_v_add_u32 v1, v2, v1                              // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v2, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v2, 0x4, v2                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16)
_v_add_u32 v1, v2, v1                              // 8. final local read offset: flrOffset = lrOffset + WOffset


/* local read addresses: final offsets a */

v_lshlrev_b32 v[vgprLocalReadAddrA], 0x4, v0       // Final Offset: offset = (lro0)*bpe


/* local read addresses: final offsets b */

v_lshlrev_b32 v[vgprLocalReadAddrB], 0x4, v1       // Final Offset: offset = (lro1)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)


/* global read addresses: tile offset assignment a */

/* LVCA = 64 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // v1 = v[vgprSerial] / 64
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
/* gro-tile *= glvw */
                                                   // v0 = v0 * 1 (multiplier is 1, do nothing)


/* global read addresses: tile offset assignment b */

/* LVCB = 8 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
v_lshrrev_b32 v2, 3, v[vgprSerial]                 // v2 = v[vgprSerial] / 8
v_and_b32 v3, 7, v[vgprSerial]                     // v3 = v[vgprSerial] % 8
/* gro-unroll *= glvw */
                                                   // v3 = v3 * 1 (multiplier is 1, do nothing)


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v1 */

/* lwaUnrollAssignmentB = v3 */


/* local write addresses: first offset a */

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v1     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x4 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x41, v3     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x4 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512*16







s_waitcnt lgkmcnt(0)                               // wait for 152 bytes of kern args
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // copy addressC
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // copy addressD
s_sub_u32 s[sgprSrdA+0], s[sgprAddressA+0], 16     // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprAddressA+1], 0     // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprSrdB+0], s[sgprAddressB+0], 16     // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprAddressB+1], 0     // pre-pad to make room for possible pointer shift

.set AddressD, UNDEF
.set AddressC, UNDEF
.set AddressA, UNDEF
.set AddressB, UNDEF

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f64 vcc, s[sgprAlpha:sgprAlpha+1], 0.0    // Alpha.real == 0.0 ?
s_cbranch_vccz label_AlphaNonZero                  // branch if Alpha.real != 0
v_cmp_eq_f64 vcc, s[sgprAlpha+2:sgprAlpha+2+1], 0.0 // Alpha.imag == 0.0 ?
s_cbranch_vccz label_AlphaNonZero                  // branch if Alpha.imag != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:



/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */
s_mov_b32 s69, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s67, s[sgprWorkGroup1], s69           // s_magic mul
s_mul_i32 s66, s[sgprWorkGroup1], s69              // s_magic mul
s_lshr_b64 s[66:67], s[66:67], 31                  // sMagicDiv
s_mul_i32 s67, s66, 8                              // quotient * non-magic divisor
s_sub_u32 s67, s[sgprWorkGroup1], s67              // WorkGroup1=remainder
s_mul_i32 s67, s67, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s67, s67, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg1
s_cmp_ge_u32 s66, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s69, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s68, s[sgprWgmRemainder1], 8         // 
s_mul_hi_u32 s3, s67, s69                          // s_magic mul
s_mul_i32 s2, s67, s69                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s68 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s67, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s66, s66, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s66 // wg1 += blockId * WGM


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */



/* global read addresses: tile offsets b */



/* global read addresses: unroll offsets a */



/* global read addresses: unroll offsets b */



/* global read addresses: branch a */



/* global read addresses: branch b */



/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 4 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideAL], 4 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x4 // scalar offset *= bytes/element


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 4 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 32 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x4 // scalar offset *= bytes/element


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s69 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
s_lshl_b64 s[68:69], s[68:69], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s69       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup1], 64            // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup1], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s69, s68, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s68, s68, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s69 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
s_lshl_b64 s[68:69], s[68:69], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s69       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)

/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 3 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 8
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

s_and_b32 s[sgprStaggerUIter], s[sgprOrigStaggerUIter], s[sgprWorkGroup0] // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], 1 // shift by StaggerUStride


/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32


/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_10                  // skip to ShadowInitStart iter b/c numIter==0


_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // G -> Reg 0_0_1_0


_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0


/* global read inc A loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_10: // 

s_mov_b32 s[sgprSrdD+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s68, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s67, s68, s[sgprStrideC1J]            // CScale s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideC1J]               // CScale s68 by Stride
s_lshl_b64 s[66:67], s[66:67], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_mul_hi_u32 s67, s68, s[sgprStrideD1J]            // Scale s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideD1J]               // Scale s68 by Stride
s_lshl_b64 s[66:67], s[66:67], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD

s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD



/* initC: remove C-tile 0-0 from pool */

/* initC: remove AB-tile 0-16 from pool */
v_accvgpr_write acc0, 0x0                          // initC
v_accvgpr_write acc1, 0x0                          // initC
v_accvgpr_write acc2, 0x0                          // initC
v_accvgpr_write acc3, 0x0                          // initC
v_accvgpr_write acc4, 0x0                          // initC
v_accvgpr_write acc5, 0x0                          // initC
v_accvgpr_write acc6, 0x0                          // initC
v_accvgpr_write acc7, 0x0                          // initC
v_accvgpr_write acc8, 0x0                          // initC
v_accvgpr_write acc9, 0x0                          // initC
v_accvgpr_write acc10, 0x0                         // initC
v_accvgpr_write acc11, 0x0                         // initC
v_accvgpr_write acc12, 0x0                         // initC
v_accvgpr_write acc13, 0x0                         // initC
v_accvgpr_write acc14, 0x0                         // initC
v_accvgpr_write acc15, 0x0                         // initC
v_accvgpr_write acc16, 0x0                         // initC
v_accvgpr_write acc17, 0x0                         // initC
v_accvgpr_write acc18, 0x0                         // initC
v_accvgpr_write acc19, 0x0                         // initC
v_accvgpr_write acc20, 0x0                         // initC
v_accvgpr_write acc21, 0x0                         // initC
v_accvgpr_write acc22, 0x0                         // initC
v_accvgpr_write acc23, 0x0                         // initC
v_accvgpr_write acc24, 0x0                         // initC
v_accvgpr_write acc25, 0x0                         // initC
v_accvgpr_write acc26, 0x0                         // initC
v_accvgpr_write acc27, 0x0                         // initC
v_accvgpr_write acc28, 0x0                         // initC
v_accvgpr_write acc29, 0x0                         // initC
v_accvgpr_write acc30, 0x0                         // initC
v_accvgpr_write acc31, 0x0                         // initC
v_accvgpr_write acc32, 0x0                         // initC
v_accvgpr_write acc33, 0x0                         // initC
v_accvgpr_write acc34, 0x0                         // initC
v_accvgpr_write acc35, 0x0                         // initC
v_accvgpr_write acc36, 0x0                         // initC
v_accvgpr_write acc37, 0x0                         // initC
v_accvgpr_write acc38, 0x0                         // initC
v_accvgpr_write acc39, 0x0                         // initC
v_accvgpr_write acc40, 0x0                         // initC
v_accvgpr_write acc41, 0x0                         // initC
v_accvgpr_write acc42, 0x0                         // initC
v_accvgpr_write acc43, 0x0                         // initC
v_accvgpr_write acc44, 0x0                         // initC
v_accvgpr_write acc45, 0x0                         // initC
v_accvgpr_write acc46, 0x0                         // initC
v_accvgpr_write acc47, 0x0                         // initC
v_accvgpr_write acc48, 0x0                         // initC
v_accvgpr_write acc49, 0x0                         // initC
v_accvgpr_write acc50, 0x0                         // initC
v_accvgpr_write acc51, 0x0                         // initC
v_accvgpr_write acc52, 0x0                         // initC
v_accvgpr_write acc53, 0x0                         // initC
v_accvgpr_write acc54, 0x0                         // initC
v_accvgpr_write acc55, 0x0                         // initC
v_accvgpr_write acc56, 0x0                         // initC
v_accvgpr_write acc57, 0x0                         // initC
v_accvgpr_write acc58, 0x0                         // initC
v_accvgpr_write acc59, 0x0                         // initC
v_accvgpr_write acc60, 0x0                         // initC
v_accvgpr_write acc61, 0x0                         // initC
v_accvgpr_write acc62, 0x0                         // initC
v_accvgpr_write acc63, 0x0                         // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_11                   // Only branch on scc1
s_getpc_B64 s[66:67]                               // addr of next instr
s_add_i32 s68, PrefetchGlobalLastIterEnd_5, 0x4    // target branch offset
s_add_u32 s66, s66, s68                            // add target branch offset
s_addc_u32 s67, s67, 0                             // add high and carry
s_setpc_b64 s[66:67]                               // branch to PrefetchGlobalLastIterEnd_5
label_NoBranch_11:

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0 8wait for global read


/* local write a */
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096

/* local write b */
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:512 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 512


/* local write swap a */


/* (EPS=1) local write swap internal offset -> 32768 */


/* local write swap b */


/* (EPS=1) local write swap internal offset -> 32768 */





/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_12:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0013: // LoopCopy1 

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-11wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //4sync for global read


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */



/* iter 0 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // G -> Reg 0_0_1_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read increment b */
/* N/A, lro->260 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // lgkmcnt=-1 vmcnt=3wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:32768 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 32768
s_waitcnt lgkmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=1 newLW=1 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */


/* iter 1 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:4160 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:4672 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:36864 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 36864
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:32768 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 32768
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:33280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 33280

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 32768 */

/* local read swap offsets b */

/* local read swap internal offset -> 32768 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=3 newLW=3 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0014: // LoopCopy2 

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-11wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //4sync for global read


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */



/* iter 0 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:33280 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // G -> Reg 0_0_1_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:33280 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */

/* local read increment b */
/* N/A, lro->260 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // lgkmcnt=-1 vmcnt=3wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt lgkmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=1 newLW=1 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */


/* iter 1 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:36864 // L -> Reg lro=256 swapByteOffset=32768 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:37376 // L -> Reg lro=256 swapByteOffset=32768 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:36928 // L -> Reg lro=260 swapByteOffset=32768 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:37440 // L -> Reg lro=260 swapByteOffset=32768 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:512 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 512

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=3 newLW=3 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */


/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
LoopEndL_evenexit_4: // unroll loop eveniter exit
s_branch LoopEndL_2                                // exit unroll loopL (and skip second exit code)
LoopEndL_oddexit_3: // unroll loop odditer exit

/* Select high bank of LDS */
v_xor_b32 v[vgprLocalReadAddrA], 0x8000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x8000, v[vgprLocalReadAddrB] // swap Red Blk
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin                                      */
/******************************************/

s_mov_b32 s66, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s66, s[sgprBeta+1], s66                   // tmp |= Beta[1] 
s_or_b32 s66, s[sgprBeta+2], s66                   // tmp |= Beta[2] 
s_or_b32 s66, s[sgprBeta+3], s66                   // tmp |= Beta[3] 
s_cmpk_eq_u32 s66, 0x0                             // Beta == 0
s_cbranch_scc0 OptNLL_End_15                       // Branch if Beta is not zero

s_mov_b32 s66, 0x00000000                          // lsb of real part of 1.0
s_mov_b32 s67, 0x3ff00000                          // msb of real part of 1.0
s_cmp_eq_u64 s[sgprAlpha:sgprAlpha+1], s[66:67]    // Alpha.real == 1.0 ?
s_cbranch_scc0 OptNLL_End_15                       // branch if alpha.real != 1
s_mov_b32 s66, 0x00000000                          // lsb of imag part of 0.0
s_mov_b32 s67, 0x00000000                          // msb of imag part of 0.0
s_cmp_eq_u64 s[sgprAlpha+2:sgprAlpha+2+1], s[66:67] // Alpha.imag == 0.0 ?
s_cbranch_scc0 OptNLL_End_15                       // branch if alpha != 1

s_and_b32 s66, 63, s[sgprSizeI]                    // s66 = s[sgprSizeI] % 64
s_add_u32 s67, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s67                // wg0 >= nwg0-1 ?
s_cselect_b32 s66, s66, 0                          // set rMT0
s_cmpk_gt_u32 s66, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required
s_and_b32 s66, 63, s[sgprSizeJ]                    // s66 = s[sgprSizeJ] % 64
s_add_u32 s67, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s67                // wg1 >= nwg1-1
s_cselect_b32 s66, s66, 0                          // set rMT1
s_cmpk_gt_u32 s66, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required

s_and_b32 s67, 7, s[sgprSizesSum+0]                // s67 = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s67, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_15                       // skip if tail loop required

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //



/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */

/* local read increment b */
/* N/A, lro->260 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */


/* iter 1 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:4160 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:4672 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */

/* Stores for OptNLL */
Summation_End_OptNLL_16:
/* endSummation: add vgpr [0...36) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ScalarGlobalReadOffsetA, UNDEF
.set ScalarGlobalReadOffsetB, UNDEF

/* Mapping of Acc register -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v1, 1, v4                            // v1 = v4 / 2
v_mul_lo_u32 v1, 0x10, v1                          // wave coordination offset 1
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
_v_add_lshl_u32 v1, v5, v1, 0                      // coordination 1 = vwb *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 4, v0                            // v0 = v0 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_and_b32 v5, 1, v4                                // v5 = v4 % 2
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 0
_v_add_lshl_u32 v0, v5, v0, 0                      // coordination 0 = vwa *(wave_id0 + tid0)
s_mul_i32 s59, 64, s[sgprWorkGroup0]               // wgp0 * MT0
_v_add_u32 v0, s59, v0                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s59, 64, s[sgprWorkGroup1]               // wgp1 * MT1
_v_add_u32 v1, s59, v1                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
GW_B0_E0_19:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=61 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
_v_add_lshl_u32 v6, v3, v0, 0x4                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc32 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc33 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc3 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc34 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc35 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc4 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc5 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc36 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc37 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc6 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc7 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc38 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc39 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+24], acc8 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc9 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc40 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc41 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc10 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc11 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc42 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc43 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc12 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc13 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc44 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc14 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc15 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc46 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc47 // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc16 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc17 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc48 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc49 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc19 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc50 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc51 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc20 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc21 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc52 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc53 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc22 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc23 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc54 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc55 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc24 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc25 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc56 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc57 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc26 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc27 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc58 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc59 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc28 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc29 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc60 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc61 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+72], acc30 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+73], acc31 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+74], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+75], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[8:11], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[12:15], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b128 v[16:19], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b128 v[20:23], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
_buffer_store_b128 v[24:27], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[28:31], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D
_buffer_store_b128 v[32:35], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D
_buffer_store_b128 v[40:43], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D
s_mul_i32 s60, s[sgprStrideD1J], 512               // scale StrideD *= numRows(32) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[44:47], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[48:51], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b128 v[52:55], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b128 v[56:59], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
_buffer_store_b128 v[60:63], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[64:67], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D
_buffer_store_b128 v[68:71], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D
_buffer_store_b128 v[72:75], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_21                           // jump to end
label_GW_End_21:

s_endpgm                                           // Kernel End
OptNLL_End_15:


/******************************************/
/* Ord. NoLoadLoop - Begin                                      */
/******************************************/


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //



/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */

/* local read increment b */
/* N/A, lro->260 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */


/* iter 1 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=256 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:4160 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:4672 // L -> Reg lro=260 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */

PrefetchGlobalLastIterEnd_5:


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */


v_and_b32 v[vgprLocalWriteAddrA], 0xf07fff, v[vgprLocalWriteAddrA] // reset to Red


/* local write reset offsets b */


v_and_b32 v[vgprLocalWriteAddrB], 0xf07fff, v[vgprLocalWriteAddrB] // reset to Red


//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 7, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_cbranch_scc1 SkipTailLoopL_8                     // skip to end of tail loop b/c numIter==0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment


/* remove stagger offsets for tail loop */

s_mov_b32 s68, 3                                   // 
s_mul_hi_u32 s67, s68, s[sgprGlobalReadIncsA+0]    // 3 * GlobalReadIncs
s_mul_i32 s66, s68, s[sgprGlobalReadIncsA+0]       // 3 * GlobalReadIncs
s_mul_hi_u32 s69, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s68, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s66, s66, s68                            // start offset S in bytes
s_subb_u32 s67, s67, s69                           // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

s_mov_b32 s68, 3                                   // 
s_mul_hi_u32 s67, s68, s[sgprGlobalReadIncsB+0]    // 3 * GlobalReadIncs
s_mul_i32 s66, s68, s[sgprGlobalReadIncsB+0]       // 3 * GlobalReadIncs
s_mul_hi_u32 s69, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s68, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s66, s66, s68                            // start offset S in bytes
s_subb_u32 s67, s67, s69                           // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
_buffer_load_b128 v[vgprG2LA+0+0:vgprG2LA+0+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 0 */
_buffer_load_b128 v[vgprG2LA+4+0:vgprG2LA+4+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // load one buffer value


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
_buffer_load_b128 v[vgprG2LB+0+0:vgprG2LB+0+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 0 */
_buffer_load_b128 v[vgprG2LB+4+0:vgprG2LB+4+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // load one buffer value

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* Done global A/B reads */




/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:512 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 512


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x7fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x7fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_6:


/* local read a */

_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read b */

_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s62, 0x1000                              // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s62, v[vgprLocalReadAddrA] // lrA += 4096 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s62, 0x1040                              // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s62, v[vgprLocalReadAddrB] // lrB += 4160 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read



/* tail loop mfma iter 0: numReadsIterCoalescedA=1, numReadsIterCoalescedB=1 */
v_and_b32 v39, 63, v[vgprSerial]                   // v39 = v[vgprSerial] % 64
v_lshrrev_b32 v39, 4, v39                          // v39 = v39 / 16
                                                   // v39 = v39 * 1 (multiplier is 1, do nothing)
v_cmp_ge_i32 s[66:67], v39, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+2], v[vgprValuB_X0_I0+4+0+0+2], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+3], v[vgprValuB_X0_I0+4+0+0+3], 0x0, s[66:67] // set 0 if K_idx >= sizeL
s_nop 1
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:7]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[32:39]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[0+0:7+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[0:7]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[32+0:39+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[32:39]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[16:23]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[48:55]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[16+0:23+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[16:23]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[48+0:55+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[48:55]
 // Ci += Ar*Bi
v_add_f64 v[40:41], -v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], 0 // Ai=-Ai
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[8:15]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[40:47]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[8+0:15+0], v[40:41], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[8:15]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[40+0:47+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1], a[40:47]
 // Ci += Ar*Bi
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[24:31]
 // Cr += Ar*Br
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[56:63]
 // Ci += Ai*Br
v_mfma_f64_16x16x4_f64 a[24+0:31+0], v[40:41], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[24:31]
 // Cr += -Ai*Bi
v_mfma_f64_16x16x4_f64 a[56+0:63+0], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1], a[56:63]
 // Ci += Ar*Bi


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x4 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x4 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_6                    // restart LoopL
TailLoopEndL_7:

SkipTailLoopL_8:

Summation_End_28:
/* endSummation: add vgpr [0...36) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ScalarGlobalReadOffsetA, UNDEF
.set ScalarGlobalReadOffsetB, UNDEF

/* Mapping of Acc register -> C Vgpr register */



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v1, 1, v4                            // v1 = v4 / 2
v_mul_lo_u32 v1, 0x10, v1                          // wave coordination offset 1
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
_v_add_lshl_u32 v1, v5, v1, 0                      // coordination 1 = vwb *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 4, v0                            // v0 = v0 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_and_b32 v5, 1, v4                                // v5 = v4 % 2
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 0
_v_add_lshl_u32 v0, v5, v0, 0                      // coordination 0 = vwa *(wave_id0 + tid0)
s_mul_i32 s59, 64, s[sgprWorkGroup0]               // wgp0 * MT0
_v_add_u32 v0, s59, v0                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s59, 64, s[sgprWorkGroup1]               // wgp1 * MT1
_v_add_u32 v1, s59, v1                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1


/* not-LocalSplitU: global write */

s_mov_b32 s59, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s59, s[sgprBeta+1], s59                   // tmp |= Beta[1] 
s_or_b32 s59, s[sgprBeta+2], s59                   // tmp |= Beta[2] 
s_or_b32 s59, s[sgprBeta+3], s59                   // tmp |= Beta[3] 
s_cmpk_eq_u32 s59, 0x0                             // Beta == 0
s_cbranch_scc0 GW_Beta_43                          // Branch if Beta is not zero

s_and_b32 s60, 63, s[sgprSizeI]                    // s60 = s[sgprSizeI] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s61                // wg0 >= nwg0-1 ?
s_cselect_b32 s60, s60, 0                          // set rMT0
s_cmpk_gt_u32 s60, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_34                         // jump if edges required
s_and_b32 s60, 63, s[sgprSizeJ]                    // s60 = s[sgprSizeJ] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s61                // wg1 >= nwg1-1
s_cselect_b32 s60, s60, 0                          // set rMT1
s_cmpk_gt_u32 s60, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_34                         // jump if edges required
GW_B0_E0_31:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=61 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
_v_add_lshl_u32 v6, v3, v0, 0x4                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc32 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc33 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc3 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc34 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc35 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc4 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc5 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc36 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc37 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc6 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc7 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc38 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc39 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+24], acc8 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc9 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc40 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc41 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc10 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc11 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc42 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc43 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc12 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc13 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc44 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc14 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc15 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc46 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc47 // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc16 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc17 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc48 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc49 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc19 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc50 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc51 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc20 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc21 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc52 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc53 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc22 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc23 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc54 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc55 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc24 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc25 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc56 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc57 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc26 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc27 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc58 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc59 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc28 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc29 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc60 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc61 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+72], acc30 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+73], acc31 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+74], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+75], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)] */
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[76:77]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[76:77]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[76:77]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[76:77]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[76:77]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[76:77]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[76:77]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[76:77]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[76:77]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[76:77]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[76:77]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[76:77]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[76:77]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[76:77]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[76:77]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[78:79]
v_mul_f64 v[76:77], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[78:79], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[76:77]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[78:79]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[8:11], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[12:15], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b128 v[16:19], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b128 v[20:23], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
_buffer_store_b128 v[24:27], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[28:31], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D
_buffer_store_b128 v[32:35], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D
_buffer_store_b128 v[40:43], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D
s_mul_i32 s60, s[sgprStrideD1J], 512               // scale StrideD *= numRows(32) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[44:47], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[48:51], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D
_buffer_store_b128 v[52:55], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
_buffer_store_b128 v[56:59], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D
_buffer_store_b128 v[60:63], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[64:67], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D
_buffer_store_b128 v[68:71], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D
_buffer_store_b128 v[72:75], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_42                           // jump to end
GW_B0_E1_34:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=48 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v6, v3, v0, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, -1, v6, s[64:65]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v4, vcc, v0, 4                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v7, v3, v4, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, -1, v7, s[64:65]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v4, vcc, v0, 8                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v16, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, -1, v16, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v4, vcc, v0, 12                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v17, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v17, -1, v17, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v4, vcc, v0, 32                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v18, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v18, -1, v18, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_v_add_co_u32 v4, vcc, v0, 36                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v19, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v19, -1, v19, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_v_add_co_u32 v4, vcc, v0, 40                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v39, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, -1, v39, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_co_u32 v4, vcc, v0, 44                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v44, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, -1, v44, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v1, vcc, v1, 32                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 32                // scale stride
_v_add_u32 v2, v2, s60                             // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 32                // scale stride
_v_add_u32 v3, v3, s60                             // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v3, v0, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v4, vcc, v0, 4                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v46, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, -1, v46, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v4, vcc, v0, 8                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v47, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, -1, v47, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v4, vcc, v0, 12                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v64, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, -1, v64, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
_v_add_co_u32 v4, vcc, v0, 32                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v65, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v65, -1, v65, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
_v_add_co_u32 v4, vcc, v0, 36                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v66, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v66, -1, v66, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
_v_add_co_u32 v4, vcc, v0, 40                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v67, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, -1, v67, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
_v_add_co_u32 v4, vcc, v0, 44                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v84, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, -1, v84, s[64:65]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc32 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc33 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc3 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc34 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc35 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc4 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc5 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc36 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc37 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc6 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc7 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc38 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc39 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc8 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc9 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc40 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc41 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc10 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc11 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc42 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc43 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc12 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc13 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc44 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+48], acc14 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+49], acc15 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+50], acc46 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+51], acc47 // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+52], acc16 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+53], acc17 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+54], acc48 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+55], acc49 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+56], acc18 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+57], acc19 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+58], acc50 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+59], acc51 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+60], acc20 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+61], acc21 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+62], acc52 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+63], acc53 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+68], acc22 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+69], acc23 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+70], acc54 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+71], acc55 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+72], acc24 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+73], acc25 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+74], acc56 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+75], acc57 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+76], acc26 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+77], acc27 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+78], acc58 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+79], acc59 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+80], acc28 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+81], acc29 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+82], acc60 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+83], acc61 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+88], acc30 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+89], acc31 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+90], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+91], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)] */
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[86:87]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[86:87]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[86:87]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[86:87]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[86:87]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[86:87]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[86:87]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[86:87]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[86:87]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[86:87]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[86:87]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[86:87]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[86:87]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+78:vgprValuC+78+1], v[86:87]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+78:vgprValuC+78+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[86:87]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[92:93]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[92:93], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[86:87]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[92:93]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[8:11], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[12:15], v7, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v16, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[24:27], v17, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[28:31], v18, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[32:35], v19, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[40:43], v39, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[48:51], v44, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v45, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[56:59], v46, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[60:63], v47, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[68:71], v64, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[72:75], v65, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[76:79], v66, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[80:83], v67, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[88:91], v84, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_42                           // jump to end
GW_Beta_43:
s_and_b32 s60, 63, s[sgprSizeI]                    // s60 = s[sgprSizeI] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s61                // wg0 >= nwg0-1 ?
s_cselect_b32 s60, s60, 0                          // set rMT0
s_cmpk_gt_u32 s60, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_41                         // jump if edges required
s_and_b32 s60, 63, s[sgprSizeJ]                    // s60 = s[sgprSizeJ] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s61                // wg1 >= nwg1-1
s_cselect_b32 s60, s60, 0                          // set rMT1
s_cmpk_gt_u32 s60, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_41                         // jump if edges required
GW_B1_E0_38:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=30 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v7, v2, v0, 0x4                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
_buffer_load_b128 v[8:11], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_buffer_load_b128 v[16:19], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_buffer_load_b128 v[24:27], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_buffer_load_b128 v[32:35], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_buffer_load_b128 v[44:47], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_buffer_load_b128 v[52:55], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:576 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_buffer_load_b128 v[60:63], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:640 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_buffer_load_b128 v[68:71], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:704 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 512               // scale StrideC *= numRows(32) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[76:79], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_buffer_load_b128 v[84:87], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:64 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_buffer_load_b128 v[92:95], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_buffer_load_b128 v[100:103], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:192 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
_buffer_load_b128 v[108:111], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
_buffer_load_b128 v[116:119], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:576 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
_buffer_load_b128 v[124:127], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:640 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
_buffer_load_b128 v[132:135], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:704 // load C for beta calc
_v_add_lshl_u32 v6, v3, v0, 0x4                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+12], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc32 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc33 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc2 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc3 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc34 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc35 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+28], acc4 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+29], acc5 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+30], acc36 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+31], acc37 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+40], acc6 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+41], acc7 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+42], acc38 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+43], acc39 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc8 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc9 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc40 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc41 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+56], acc10 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+57], acc11 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+58], acc42 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+59], acc43 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+64], acc12 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+65], acc13 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+66], acc44 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+67], acc45 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+72], acc14 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+73], acc15 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+74], acc46 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+75], acc47 // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+80], acc16 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+81], acc17 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+82], acc48 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+83], acc49 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+88], acc18 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+89], acc19 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+90], acc50 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+91], acc51 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+96], acc20 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+97], acc21 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+98], acc52 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+99], acc53 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+104], acc22 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+105], acc23 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+106], acc54 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+107], acc55 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+112], acc24 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+113], acc25 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+114], acc56 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+115], acc57 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+120], acc26 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+121], acc27 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+122], acc58 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+123], acc59 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+128], acc28 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+129], acc29 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+130], acc60 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+131], acc61 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+136], acc30 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+137], acc31 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+138], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+139], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)] */
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[140:141]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[140:141]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[140:141]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[140:141]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[140:141]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[140:141]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[140:141]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[140:141]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[140:141]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[140:141]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[140:141]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[140:141]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+114:vgprValuC+114+1], v[140:141]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+114:vgprValuC+114+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+122:vgprValuC+122+1], v[140:141]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+122:vgprValuC+122+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+128:vgprValuC+128+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+128:vgprValuC+128+1] // 
v_fma_f64 v[vgprValuC+128:vgprValuC+128+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+130:vgprValuC+130+1], v[140:141]
v_fma_f64 v[vgprValuC+130:vgprValuC+130+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+130:vgprValuC+130+1], v[142:143]
v_mul_f64 v[140:141], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+136:vgprValuC+136+1] // 
v_mul_f64 v[142:143], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+136:vgprValuC+136+1] // 
v_fma_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+138:vgprValuC+138+1], v[140:141]
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+138:vgprValuC+138+1], v[142:143]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[8:9], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[10:11], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[8:9], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[10:11], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 1 + 1 - 1
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[16:17], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[18:19], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[16:17], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[18:19], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 2 + 2 - 1
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[24:25], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[26:27], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[24:25], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+30:vgprValuC+30+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[26:27], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+30:vgprValuC+30+1]
_buffer_store_b128 v[28:31], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 3 + 3 - 1
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[32:33], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[34:35], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[32:33], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+42:vgprValuC+42+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[34:35], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+42:vgprValuC+42+1]
_buffer_store_b128 v[40:43], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 4 + 4 - 1
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[44:45], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[46:47], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[44:45], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+50:vgprValuC+50+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[46:47], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+50:vgprValuC+50+1]
_buffer_store_b128 v[48:51], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 5 + 5 - 1
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[52:53], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[54:55], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[52:53], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+58:vgprValuC+58+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[54:55], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+58:vgprValuC+58+1]
_buffer_store_b128 v[56:59], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 6 + 6 - 1
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[60:61], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[62:63], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[60:61], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+66:vgprValuC+66+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[62:63], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+66:vgprValuC+66+1]
_buffer_store_b128 v[64:67], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 7 + 7 - 1
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[68:69], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[70:71], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[68:69], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+74:vgprValuC+74+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[70:71], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+74:vgprValuC+74+1]
_buffer_store_b128 v[72:75], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 8 + 8 - 1
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[76:77], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[78:79], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[76:77], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+82:vgprValuC+82+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[78:79], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+82:vgprValuC+82+1]
s_mul_i32 s60, s[sgprStrideD1J], 512               // scale StrideD *= numRows(32) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[80:83], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 9 + 9 - 1
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[84:85], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[86:87], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[84:85], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+90:vgprValuC+90+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[86:87], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+90:vgprValuC+90+1]
_buffer_store_b128 v[88:91], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:64 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 10 + 10 - 1
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[92:93], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[94:95], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[92:93], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+98:vgprValuC+98+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[94:95], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+98:vgprValuC+98+1]
_buffer_store_b128 v[96:99], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 11 + 11 - 1
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[100:101], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[102:103], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[100:101], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+106:vgprValuC+106+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[102:103], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+106:vgprValuC+106+1]
_buffer_store_b128 v[104:107], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:192 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 12 + 12 - 1
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[108:109], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[110:111], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[108:109], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+114:vgprValuC+114+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[110:111], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+114:vgprValuC+114+1]
_buffer_store_b128 v[112:115], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 13 + 13 - 1
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[116:117], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[118:119], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[116:117], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+122:vgprValuC+122+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[118:119], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+122:vgprValuC+122+1]
_buffer_store_b128 v[120:123], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:576 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 14 + 14 - 1
v_fma_f64 v[vgprValuC+128:vgprValuC+128+1], v[124:125], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+128:vgprValuC+128+1]
v_fma_f64 v[vgprValuC+128:vgprValuC+128+1], v[126:127], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+128:vgprValuC+128+1]
v_fma_f64 v[vgprValuC+130:vgprValuC+130+1], v[124:125], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+130:vgprValuC+130+1]
v_fma_f64 v[vgprValuC+130:vgprValuC+130+1], v[126:127], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+130:vgprValuC+130+1]
_buffer_store_b128 v[128:131], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:640 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 15 + 15 - 1
v_fma_f64 v[vgprValuC+136:vgprValuC+136+1], v[132:133], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+136:vgprValuC+136+1]
v_fma_f64 v[vgprValuC+136:vgprValuC+136+1], v[134:135], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+136:vgprValuC+136+1]
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[132:133], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+138:vgprValuC+138+1]
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[134:135], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+138:vgprValuC+138+1]
_buffer_store_b128 v[136:139], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:704 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_42                           // jump to end
GW_B1_E1_41:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=27 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v6, v2, v0, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, -1, v6, s[64:65]                 // LDC clip if OOB. offset
_buffer_load_b128 v[8:11], v6, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v6, v3, v0, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, -1, v6, s[64:65]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v4, vcc, v0, 4                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v7, v2, v4, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, -1, v7, s[64:65]                 // LDC clip if OOB. offset
_buffer_load_b128 v[16:19], v7, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v7, v3, v4, 0x4                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, -1, v7, s[64:65]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v4, vcc, v0, 8                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v24, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, -1, v24, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[28:31], v24, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v24, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, -1, v24, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v4, vcc, v0, 12                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v25, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, -1, v25, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[40:43], v25, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v25, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, -1, v25, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v4, vcc, v0, 32                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v26, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, -1, v26, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[48:51], v26, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v26, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, -1, v26, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_v_add_co_u32 v4, vcc, v0, 36                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v27, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, -1, v27, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[56:59], v27, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v27, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, -1, v27, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_v_add_co_u32 v4, vcc, v0, 40                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v39, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, -1, v39, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[64:67], v39, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v39, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, -1, v39, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_co_u32 v4, vcc, v0, 44                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v72, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[76:79], v72, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v72, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v1, vcc, v1, 32                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 32                // scale stride
_v_add_u32 v2, v2, s60                             // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 32                // scale stride
_v_add_u32 v3, v3, s60                             // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v73, v2, v0, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[84:87], v73, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v73, v3, v0, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v4, vcc, v0, 4                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v74, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[92:95], v74, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v74, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v4, vcc, v0, 8                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v75, v2, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[100:103], v75, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v75, v3, v4, 0x4                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v4, vcc, v0, 12                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v108, v2, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v108, -1, v108, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[112:115], v108, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v108, v3, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v108, -1, v108, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
_v_add_co_u32 v4, vcc, v0, 32                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v109, v2, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v109, -1, v109, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[120:123], v109, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v109, v3, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v109, -1, v109, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
_v_add_co_u32 v4, vcc, v0, 36                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v110, v2, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v110, -1, v110, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[128:131], v110, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v110, v3, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v110, -1, v110, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
_v_add_co_u32 v4, vcc, v0, 40                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v111, v2, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v111, -1, v111, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[136:139], v111, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v111, v3, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v111, -1, v111, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
_v_add_co_u32 v4, vcc, v0, 44                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[64:65], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v2, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[148:151], v144, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v3, v4, 0x4                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+12], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc32 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc33 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc2 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc3 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc34 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc35 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+32], acc4 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+33], acc5 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+34], acc36 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+35], acc37 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc6 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc38 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc39 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+52], acc8 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+53], acc9 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+54], acc40 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+55], acc41 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+60], acc10 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+61], acc11 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+62], acc42 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+63], acc43 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+68], acc12 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+69], acc13 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+70], acc44 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+71], acc45 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+80], acc14 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+81], acc15 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+82], acc46 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+83], acc47 // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+88], acc16 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+89], acc17 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+90], acc48 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+91], acc49 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+96], acc18 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+97], acc19 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+98], acc50 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+99], acc51 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+104], acc20 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+105], acc21 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+106], acc52 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+107], acc53 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+116], acc22 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+117], acc23 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+118], acc54 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+119], acc55 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+124], acc24 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+125], acc25 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+126], acc56 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+127], acc57 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+132], acc26 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+133], acc27 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+134], acc58 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+135], acc59 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+140], acc28 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+141], acc29 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+142], acc60 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+143], acc61 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+152], acc30 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+153], acc31 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+154], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+155], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)] */
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[146:147]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[146:147]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[146:147]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[146:147]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[146:147]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[146:147]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[146:147]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[146:147]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[146:147]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[146:147]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[146:147]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+118:vgprValuC+118+1], v[146:147]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+118:vgprValuC+118+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+126:vgprValuC+126+1], v[146:147]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+126:vgprValuC+126+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+132:vgprValuC+132+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+132:vgprValuC+132+1] // 
v_fma_f64 v[vgprValuC+132:vgprValuC+132+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+134:vgprValuC+134+1], v[146:147]
v_fma_f64 v[vgprValuC+134:vgprValuC+134+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+134:vgprValuC+134+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+140:vgprValuC+140+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+140:vgprValuC+140+1] // 
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+142:vgprValuC+142+1], v[146:147]
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+142:vgprValuC+142+1], v[156:157]
v_mul_f64 v[146:147], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+152:vgprValuC+152+1] // 
v_mul_f64 v[156:157], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+152:vgprValuC+152+1] // 
v_fma_f64 v[vgprValuC+152:vgprValuC+152+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+154:vgprValuC+154+1], v[146:147]
v_fma_f64 v[vgprValuC+154:vgprValuC+154+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+154:vgprValuC+154+1], v[156:157]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[8:9], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[10:11], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[8:9], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[10:11], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v6, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[16:17], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[18:19], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[16:17], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[18:19], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v7, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[28:29], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[30:31], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[28:29], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+34:vgprValuC+34+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[30:31], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+34:vgprValuC+34+1]
_buffer_store_b128 v[32:35], v24, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[40:41], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[42:43], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[40:41], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+46:vgprValuC+46+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[42:43], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+46:vgprValuC+46+1]
_buffer_store_b128 v[44:47], v25, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[48:49], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[50:51], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[48:49], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+54:vgprValuC+54+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[50:51], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+54:vgprValuC+54+1]
_buffer_store_b128 v[52:55], v26, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[56:57], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[58:59], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[56:57], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+62:vgprValuC+62+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[58:59], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+62:vgprValuC+62+1]
_buffer_store_b128 v[60:63], v27, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[64:65], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[66:67], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[64:65], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+70:vgprValuC+70+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[66:67], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+70:vgprValuC+70+1]
_buffer_store_b128 v[68:71], v39, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[76:77], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[78:79], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[76:77], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+82:vgprValuC+82+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[78:79], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+82:vgprValuC+82+1]
_buffer_store_b128 v[80:83], v72, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[84:85], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[86:87], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[84:85], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+90:vgprValuC+90+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[86:87], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+90:vgprValuC+90+1]
_buffer_store_b128 v[88:91], v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[92:93], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[94:95], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[92:93], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+98:vgprValuC+98+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[94:95], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+98:vgprValuC+98+1]
_buffer_store_b128 v[96:99], v74, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[100:101], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[102:103], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[100:101], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+106:vgprValuC+106+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[102:103], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+106:vgprValuC+106+1]
_buffer_store_b128 v[104:107], v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[112:113], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[114:115], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[112:113], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+118:vgprValuC+118+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[114:115], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+118:vgprValuC+118+1]
_buffer_store_b128 v[116:119], v108, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[120:121], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[122:123], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[120:121], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+126:vgprValuC+126+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[122:123], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+126:vgprValuC+126+1]
_buffer_store_b128 v[124:127], v109, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+132:vgprValuC+132+1], v[128:129], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+132:vgprValuC+132+1]
v_fma_f64 v[vgprValuC+132:vgprValuC+132+1], v[130:131], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+132:vgprValuC+132+1]
v_fma_f64 v[vgprValuC+134:vgprValuC+134+1], v[128:129], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+134:vgprValuC+134+1]
v_fma_f64 v[vgprValuC+134:vgprValuC+134+1], v[130:131], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+134:vgprValuC+134+1]
_buffer_store_b128 v[132:135], v110, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+140:vgprValuC+140+1]
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+140:vgprValuC+140+1]
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+142:vgprValuC+142+1]
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+142:vgprValuC+142+1]
_buffer_store_b128 v[140:143], v111, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+152:vgprValuC+152+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+152:vgprValuC+152+1]
v_fma_f64 v[vgprValuC+152:vgprValuC+152+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+152:vgprValuC+152+1]
v_fma_f64 v[vgprValuC+154:vgprValuC+154+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+154:vgprValuC+154+1]
v_fma_f64 v[vgprValuC+154:vgprValuC+154+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+154:vgprValuC+154+1]
_buffer_store_b128 v[152:155], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_42                           // jump to end
label_GW_End_42:

label_0047:  /// KernelEnd
s_endpgm                                           // Kernel End

