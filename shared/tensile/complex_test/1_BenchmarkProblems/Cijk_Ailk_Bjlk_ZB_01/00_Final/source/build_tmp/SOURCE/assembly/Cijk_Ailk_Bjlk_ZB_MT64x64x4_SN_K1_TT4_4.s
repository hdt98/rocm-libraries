
/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureDefault
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4
.globl Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4
.p2align 8
.type Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_user_sgpr_kernarg_preload_length 0
  .amdhsa_user_sgpr_count 2
  .amdhsa_accum_offset 144 // accvgpr offset
  .amdhsa_next_free_vgpr 144 // vgprs
  .amdhsa_next_free_sgpr 69 // sgprs
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
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
/* ThreadTile= 4 x 4 */
/* SubGroup= 16 x 16 */
/* VectorWidth=1 */
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
  - .name: Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4
    .symbol: 'Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4.kd'
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
    .group_segment_fixed_size:   16384
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 69
    .sgpr_spill_count:           0
    .vgpr_count:                 143
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4:

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
/* ValuC range: [0-64),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 64
.set vgprValuA_X1_I0, 80
.set vgprG2LA, 132
.set vgprValuB_X0_I0, 96
.set vgprValuB_X1_I0, 112
.set vgprG2LB, 136
.set vgprLocalWriteAddrA, 128
.set vgprLocalWriteAddrB, 129
.set vgprGlobalReadOffsetA, 130
.set vgprGlobalReadOffsetB, 131
.set vgprLocalReadAddrA, 140
.set vgprLocalReadAddrB, 141
.set vgprSerial, 142
/* Num VGPR=143 */
/* Num AccVGPR=0 */

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
/* max SGPR=69 */

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
.set constStrideB1J, 1
.set sgprStrideBL, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 64
.set MT1, 64
.set DepthU, 4
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
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffset1J:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideBL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset1J], v[\vgprTmp+0] // accumulate K lower
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
/* 4x4 thread-tile                        */
/******************************************/
.macro MAC_4x4_X0
// Component.MAC.FMA_F64C_Plain
v_fma_f64 v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1]
s_setprio 1 // Raise priority while processing macs
v_fma_f64 v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1]
v_fma_f64 v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1]
v_fma_f64 v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1]
v_fma_f64 v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1]
v_fma_f64 v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1]
v_fma_f64 v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1]
v_fma_f64 v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1]
v_fma_f64 v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1]
v_fma_f64 v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1]
v_fma_f64 v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1]
v_fma_f64 v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1]
v_fma_f64 v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1]
v_fma_f64 v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1]
v_fma_f64 v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1]
v_fma_f64 v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1]
v_fma_f64 v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1]
v_fma_f64 v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1]
v_fma_f64 v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1]
v_fma_f64 v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1]
v_fma_f64 v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1]
v_fma_f64 v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1]
v_fma_f64 v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1]
v_fma_f64 v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1]
v_fma_f64 v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1]
v_fma_f64 v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1]
v_fma_f64 v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1]
v_fma_f64 v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1]
v_fma_f64 v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1]
v_fma_f64 v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1]
v_fma_f64 v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3]
v_fma_f64 v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3]
v_fma_f64 v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3]
v_fma_f64 v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3]
v_fma_f64 v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3]
v_fma_f64 v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3]
v_fma_f64 v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3]
v_fma_f64 v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3]
v_fma_f64 v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3]
v_fma_f64 v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3]
v_fma_f64 v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3]
v_fma_f64 v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3]
v_fma_f64 v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3]
v_fma_f64 v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3]
v_fma_f64 v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3]
v_fma_f64 v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3]
v_fma_f64 v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3]
v_fma_f64 v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3]
v_fma_f64 v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3]
v_fma_f64 v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+0*4+0:vgprValuB_X0_I0+0*4+1], v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3]
v_fma_f64 v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3]
v_fma_f64 v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3]
v_fma_f64 v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3]
v_fma_f64 v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+1*4+0:vgprValuB_X0_I0+1*4+1], v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3]
v_fma_f64 v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3]
v_fma_f64 v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3]
v_fma_f64 v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3]
v_fma_f64 v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+2*4+0:vgprValuB_X0_I0+2*4+1], v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3]
v_fma_f64 v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3]
v_fma_f64 v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3]
v_fma_f64 v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3]
v_fma_f64 v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+3*4+0:vgprValuB_X0_I0+3*4+1], v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3]
s_setprio 0 // Reset priority after macs
.endm
.macro MAC_4x4_X1
// Component.MAC.FMA_F64C_Plain
v_fma_f64 v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1]
s_setprio 1 // Raise priority while processing macs
v_fma_f64 v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1]
v_fma_f64 v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1]
v_fma_f64 v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1]
v_fma_f64 v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1]
v_fma_f64 v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1]
v_fma_f64 v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1]
v_fma_f64 v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1]
v_fma_f64 v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1]
v_fma_f64 v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1]
v_fma_f64 v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1]
v_fma_f64 v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1]
v_fma_f64 v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1]
v_fma_f64 v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1]
v_fma_f64 v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1]
v_fma_f64 v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], -v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1]
v_fma_f64 v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], -v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1]
v_fma_f64 v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], -v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1]
v_fma_f64 v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], -v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1]
v_fma_f64 v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], -v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1]
v_fma_f64 v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], -v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1]
v_fma_f64 v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], -v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1]
v_fma_f64 v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], -v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1]
v_fma_f64 v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], -v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1]
v_fma_f64 v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], -v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1]
v_fma_f64 v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], -v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1]
v_fma_f64 v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], -v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1]
v_fma_f64 v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], -v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1]
v_fma_f64 v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], -v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1]
v_fma_f64 v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], -v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1]
v_fma_f64 v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], -v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3]
v_fma_f64 v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3]
v_fma_f64 v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3]
v_fma_f64 v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+0*4+2:vgprValuB_X1_I0+0*4+3], v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3]
v_fma_f64 v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3]
v_fma_f64 v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3]
v_fma_f64 v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3]
v_fma_f64 v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+1*4+2:vgprValuB_X1_I0+1*4+3], v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3]
v_fma_f64 v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3]
v_fma_f64 v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3]
v_fma_f64 v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3]
v_fma_f64 v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+2*4+2:vgprValuB_X1_I0+2*4+3], v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3]
v_fma_f64 v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3], v[vgprValuA_X1_I0+0*4+0:vgprValuA_X1_I0+0*4+1], v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3]
v_fma_f64 v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3], v[vgprValuA_X1_I0+1*4+0:vgprValuA_X1_I0+1*4+1], v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3]
v_fma_f64 v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3], v[vgprValuA_X1_I0+2*4+0:vgprValuA_X1_I0+2*4+1], v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3]
v_fma_f64 v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3], v[vgprValuA_X1_I0+3*4+0:vgprValuA_X1_I0+3*4+1], v[vgprValuB_X1_I0+3*4+2:vgprValuB_X1_I0+3*4+3], v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3]
v_fma_f64 v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3]
v_fma_f64 v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3]
v_fma_f64 v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3]
v_fma_f64 v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], v[vgprValuB_X1_I0+0*4+0:vgprValuB_X1_I0+0*4+1], v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3]
v_fma_f64 v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3]
v_fma_f64 v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3]
v_fma_f64 v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3]
v_fma_f64 v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], v[vgprValuB_X1_I0+1*4+0:vgprValuB_X1_I0+1*4+1], v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3]
v_fma_f64 v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3]
v_fma_f64 v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3]
v_fma_f64 v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3]
v_fma_f64 v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], v[vgprValuB_X1_I0+2*4+0:vgprValuB_X1_I0+2*4+1], v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3]
v_fma_f64 v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3], v[vgprValuA_X1_I0+0*4+2:vgprValuA_X1_I0+0*4+3], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3]
v_fma_f64 v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3], v[vgprValuA_X1_I0+1*4+2:vgprValuA_X1_I0+1*4+3], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3]
v_fma_f64 v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3], v[vgprValuA_X1_I0+2*4+2:vgprValuA_X1_I0+2*4+3], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3]
v_fma_f64 v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3], v[vgprValuA_X1_I0+3*4+2:vgprValuA_X1_I0+3*4+3], v[vgprValuB_X1_I0+3*4+0:vgprValuB_X1_I0+3*4+1], v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3]
s_setprio 0 // Reset priority after macs
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

Cijk_Ailk_Bjlk_ZB_MT64x64x4_SN_K1_TT4_4_preloaded: // Kernel start when preloading

/* Load Kernel Args */
_s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 // 
_s_load_b512 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
_s_load_b128 s[56:59], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // 
_s_load_b64 s[60:61], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // 
s_mov_b32 m0, 0x4000                               // LDS clamp at 16384 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

/*lr0I = serial % SG0I*/
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // v1 = v[vgprSerial] % 16
/*lr1J = (serial / SG1J) % SG1J*/
v_lshrrev_b32 v2, 4, v0                            // v2 = v0 / 16
v_and_b32 v3, 15, v0                               // v3 = v0 % 16


/* local read addresses: final offsets a */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
v_lshlrev_b32 v0, 0x6, v0                          // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
_v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x4 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
v_lshlrev_b32 v0, 0x6, v0                          // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x4 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x1000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)


/* global read addresses: tile offset assignment a */

/* LVCA = 64 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // v1 = v[vgprSerial] / 64
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
/* gro-tile *= glvw */
                                                   // v0 = v0 * 1 (multiplier is 1, do nothing)


/* global read addresses: tile offset assignment b */

/* LVCB = 64 */
/* v2 = (local)groB-tile = serial%LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial/LVCB */
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // v3 = v[vgprSerial] / 64
v_and_b32 v2, 63, v[vgprSerial]                    // v2 = v[vgprSerial] % 64
/* gro-tile *= glvw */
                                                   // v2 = v2 * 1 (multiplier is 1, do nothing)


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

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v3     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x4 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x1000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=256*16







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
s_mov_b32 s67, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s65, s[sgprWorkGroup1], s67           // s_magic mul
s_mul_i32 s64, s[sgprWorkGroup1], s67              // s_magic mul
s_lshr_b64 s[64:65], s[64:65], 31                  // sMagicDiv
s_mul_i32 s65, s64, 8                              // quotient * non-magic divisor
s_sub_u32 s65, s[sgprWorkGroup1], s65              // WorkGroup1=remainder
s_mul_i32 s65, s65, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s65, s65, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg1
s_cmp_ge_u32 s64, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s67, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s66, s[sgprWgmRemainder1], 8         // 
s_mul_hi_u32 s3, s65, s67                          // s_magic mul
s_mul_i32 s2, s65, s67                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s66 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s65, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s64, s64, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s64 // wg1 += blockId * WGM


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



/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 4 // gROA_0_0_0_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  2,  3, 4 // gROB_0_0_0_0


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s67, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s66, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s66 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s67 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32
s_mul_hi_u32 s65, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s64, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s66, s66, s64                            // accum wg term to tilestart
s_addc_u32 s67, s67, s65                           // accum wg term to tilestart
s_lshl_b64 s[66:67], s[66:67], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s67, s[sgprWorkGroup1], 64            // WorkGroup[01] * MT
s_mul_i32 s66, s[sgprWorkGroup1], 64               // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s66 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s67 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_mul_hi_u32 s65, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s64, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s66, s66, s64                            // accum wg term to tilestart
s_addc_u32 s67, s67, s65                           // accum wg term to tilestart
s_lshl_b64 s[66:67], s[66:67], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mul_i32 s[sgprGlobalReadIncsB+0], DepthU*BpeB, s[sgprStrideBL] // incrB unrollIdx)

/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 2 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 4
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

s_and_b32 s[sgprStaggerUIter], s[sgprOrigStaggerUIter], s[sgprWorkGroup0] // Compute actual stagger start for this tile


/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s65      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32


/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s65      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
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


_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0


/* global read inc A loopL */
s_add_u32 s66, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s66              // Is this wrapIter? (pf)
s_cselect_b32 s64, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s65      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s66, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s66              // Is this wrapIter? (pf)
s_cselect_b32 s64, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s65      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
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


s_mul_i32 s64, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s63, s64, s[sgprStrideC1J]            // CScale s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideC1J]               // CScale s64 by Stride
s_lshl_b64 s[62:63], s[62:63], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63       // add hi to SRD
s_mul_hi_u32 s63, s64, s[sgprStrideD1J]            // Scale s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideD1J]               // Scale s64 by Stride
s_lshl_b64 s[62:63], s[62:63], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi to SRD

s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63       // add hi to SRD
s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi to SRD



/* initC: remove C-tile 0-64 from pool */

/* initC: remove AB-tile 64-128 from pool */
v_mov_b64 v[vgprValuC+0:vgprValuC+0+1], 0x0        // initC
v_mov_b64 v[vgprValuC+2:vgprValuC+2+1], 0x0        // initC
v_mov_b64 v[vgprValuC+4:vgprValuC+4+1], 0x0        // initC
v_mov_b64 v[vgprValuC+6:vgprValuC+6+1], 0x0        // initC
v_mov_b64 v[vgprValuC+8:vgprValuC+8+1], 0x0        // initC
v_mov_b64 v[vgprValuC+10:vgprValuC+10+1], 0x0      // initC
v_mov_b64 v[vgprValuC+12:vgprValuC+12+1], 0x0      // initC
v_mov_b64 v[vgprValuC+14:vgprValuC+14+1], 0x0      // initC
v_mov_b64 v[vgprValuC+16:vgprValuC+16+1], 0x0      // initC
v_mov_b64 v[vgprValuC+18:vgprValuC+18+1], 0x0      // initC
v_mov_b64 v[vgprValuC+20:vgprValuC+20+1], 0x0      // initC
v_mov_b64 v[vgprValuC+22:vgprValuC+22+1], 0x0      // initC
v_mov_b64 v[vgprValuC+24:vgprValuC+24+1], 0x0      // initC
v_mov_b64 v[vgprValuC+26:vgprValuC+26+1], 0x0      // initC
v_mov_b64 v[vgprValuC+28:vgprValuC+28+1], 0x0      // initC
v_mov_b64 v[vgprValuC+30:vgprValuC+30+1], 0x0      // initC
v_mov_b64 v[vgprValuC+32:vgprValuC+32+1], 0x0      // initC
v_mov_b64 v[vgprValuC+34:vgprValuC+34+1], 0x0      // initC
v_mov_b64 v[vgprValuC+36:vgprValuC+36+1], 0x0      // initC
v_mov_b64 v[vgprValuC+38:vgprValuC+38+1], 0x0      // initC
v_mov_b64 v[vgprValuC+40:vgprValuC+40+1], 0x0      // initC
v_mov_b64 v[vgprValuC+42:vgprValuC+42+1], 0x0      // initC
v_mov_b64 v[vgprValuC+44:vgprValuC+44+1], 0x0      // initC
v_mov_b64 v[vgprValuC+46:vgprValuC+46+1], 0x0      // initC
v_mov_b64 v[vgprValuC+48:vgprValuC+48+1], 0x0      // initC
v_mov_b64 v[vgprValuC+50:vgprValuC+50+1], 0x0      // initC
v_mov_b64 v[vgprValuC+52:vgprValuC+52+1], 0x0      // initC
v_mov_b64 v[vgprValuC+54:vgprValuC+54+1], 0x0      // initC
v_mov_b64 v[vgprValuC+56:vgprValuC+56+1], 0x0      // initC
v_mov_b64 v[vgprValuC+58:vgprValuC+58+1], 0x0      // initC
v_mov_b64 v[vgprValuC+60:vgprValuC+60+1], 0x0      // initC
v_mov_b64 v[vgprValuC+62:vgprValuC+62+1], 0x0      // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_11                   // Only branch on scc1
s_getpc_B64 s[62:63]                               // addr of next instr
s_add_i32 s64, PrefetchGlobalLastIterEnd_5, 0x4    // target branch offset
s_add_u32 s62, s62, s64                            // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_setpc_b64 s[62:63]                               // branch to PrefetchGlobalLastIterEnd_5
label_NoBranch_11:

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0 8wait for global read


/* local write a */
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local write b */
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0


/* local write swap a */


/* (EPS=1) local write swap internal offset -> 8192 */


/* local write swap b */


/* (EPS=1) local write swap internal offset -> 8192 */



s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-10prefetch wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* local read prefetch a */

_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read prefetch b */

_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */


/* local read inc b */

/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */



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


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */



/* iter 0 */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s62, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s63, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s62, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s63, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 1 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:8192 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 8192
s_waitcnt lgkmcnt(9)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:8192 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 8192

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 8192 */

/* local read swap offsets b */

/* local read swap internal offset -> 8192 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(9)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X0

/* iter 3 */

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //

/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:8192 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:8192 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:8448 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:8704 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:8960 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1

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


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */



/* iter 0 */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:9472 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:9728 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:9984 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s62, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s63, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s62, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s63, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:9472 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:9728 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:9984 // L -> Reg lro=64 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 1 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:10240 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:10496 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:10752 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:11008 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:10496 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:10752 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:11008 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt lgkmcnt(9)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:11264 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:11520 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:11776 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:12032 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:11264 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:11520 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:11776 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:12032 // L -> Reg lro=192 swapByteOffset=8192 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(9)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X0

/* iter 3 */

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //

/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1

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
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin                                      */
/******************************************/

s_mov_b32 s62, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s62, s[sgprBeta+1], s62                   // tmp |= Beta[1] 
s_or_b32 s62, s[sgprBeta+2], s62                   // tmp |= Beta[2] 
s_or_b32 s62, s[sgprBeta+3], s62                   // tmp |= Beta[3] 
s_cmpk_eq_u32 s62, 0x0                             // Beta == 0
s_cbranch_scc0 OptNLL_End_15                       // Branch if Beta is not zero

s_mov_b32 s62, 0x00000000                          // lsb of real part of 1.0
s_mov_b32 s63, 0x3ff00000                          // msb of real part of 1.0
s_cmp_eq_u64 s[sgprAlpha:sgprAlpha+1], s[62:63]    // Alpha.real == 1.0 ?
s_cbranch_scc0 OptNLL_End_15                       // branch if alpha.real != 1
s_mov_b32 s62, 0x00000000                          // lsb of imag part of 0.0
s_mov_b32 s63, 0x00000000                          // msb of imag part of 0.0
s_cmp_eq_u64 s[sgprAlpha+2:sgprAlpha+2+1], s[62:63] // Alpha.imag == 0.0 ?
s_cbranch_scc0 OptNLL_End_15                       // branch if alpha != 1

s_and_b32 s62, 63, s[sgprSizeI]                    // s62 = s[sgprSizeI] % 64
s_add_u32 s63, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s63                // wg0 >= nwg0-1 ?
s_cselect_b32 s62, s62, 0                          // set rMT0
s_cmpk_gt_u32 s62, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required
s_and_b32 s62, 63, s[sgprSizeJ]                    // s62 = s[sgprSizeJ] % 64
s_add_u32 s63, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s63                // wg1 >= nwg1-1
s_cselect_b32 s62, s62, 0                          // set rMT1
s_cmpk_gt_u32 s62, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_15                       // jump if edges required

s_and_b32 s63, 3, s[sgprSizesSum+0]                // s63 = s[sgprSizesSum+0] % 4
s_cmp_eq_u32 s63, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_15                       // skip if tail loop required



/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 1 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1

/* iter 2 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 3 (last unrolled loop) */

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x4_X1
/* Stores for OptNLL */
Summation_End_OptNLL_16:
/* endSummation: add vgpr [64...140) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
/* computeStoreVgprs */
v_lshrrev_b32 v65, 4, v[vgprSerial]                // v65 = v[vgprSerial] / 16
v_and_b32 v64, 15, v[vgprSerial]                   // v64 = v[vgprSerial] % 16
                                                   // v64 = v64 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            // rowStart vgpr
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            // rowStart vgpr

s_mul_i32 s60, 0x40, s[sgprWorkGroup0]             // s60 = wg0*MT0
_v_add_co_u32 v64, vcc, s60, v64                   // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s62, 0x40, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v65, vcc, s62, v65                   // coord1 = tid1*VW + wg1*MT1
GW_B0_E0_19:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_lshl_u32 v70, v67, v64, 0x4                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[0:3], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[4:7], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[8:11], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[12:15], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[16:19], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[24:27], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[28:31], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[32:35], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[36:39], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[40:43], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[44:47], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[48:51], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[56:59], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[60:63], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_branch label_GW_End_21                           // jump to end
label_GW_End_21:

s_endpgm                                           // Kernel End
OptNLL_End_15:


/******************************************/
/* Ord. NoLoadLoop - Begin                                      */
/******************************************/




/* iter 0 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 1 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1

/* iter 2 (last unrolled loop) */


/* local read a */
_ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=192 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0

/* iter 3 (last unrolled loop) */

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x4_X1
PrefetchGlobalLastIterEnd_5:


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */


v_and_b32 v[vgprLocalWriteAddrA], 0xf01fff, v[vgprLocalWriteAddrA] // reset to Red


/* local write reset offsets b */


v_and_b32 v[vgprLocalWriteAddrB], 0xf01fff, v[vgprLocalWriteAddrB] // reset to Red


//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 3, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 4
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_cbranch_scc1 SkipTailLoopL_8                     // skip to end of tail loop b/c numIter==0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment


/* remove stagger offsets for tail loop */

s_mov_b32 s64, 3                                   // 
s_mul_hi_u32 s63, s64, s[sgprGlobalReadIncsA+0]    // 3 * GlobalReadIncs
s_mul_i32 s62, s64, s[sgprGlobalReadIncsA+0]       // 3 * GlobalReadIncs
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s62, s62, s64                            // start offset S in bytes
s_subb_u32 s63, s63, s65                           // start offset S in bytes
s_sub_u32 s62, s62, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s63, s63, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

s_mov_b32 s64, 3                                   // 
s_mul_hi_u32 s63, s64, s[sgprGlobalReadIncsB+0]    // 3 * GlobalReadIncs
s_mul_i32 s62, s64, s[sgprGlobalReadIncsB+0]       // 3 * GlobalReadIncs
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s62, s62, s64                            // start offset S in bytes
s_subb_u32 s63, s63, s65                           // start offset S in bytes
s_sub_u32 s62, s62, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s63, s63, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s62        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s63      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
_buffer_load_b128 v[vgprG2LA+0+0:vgprG2LA+0+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
_buffer_load_b128 v[vgprG2LB+0+0:vgprG2LB+0+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* Done global A/B reads */




/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x1fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x1fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_6:


/* local read a */

_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read b */

_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s62, 0x400                               // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s62, v[vgprLocalReadAddrA] // lrA += 1024 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s62, 0x400                               // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s62, v[vgprLocalReadAddrB] // lrB += 1024 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read

MAC_4x4_X0

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_6                    // restart LoopL
TailLoopEndL_7:

SkipTailLoopL_8:

Summation_End_28:
/* endSummation: add vgpr [64...140) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v65, 4, v[vgprSerial]                // v65 = v[vgprSerial] / 16
v_and_b32 v64, 15, v[vgprSerial]                   // v64 = v[vgprSerial] % 16
                                                   // v64 = v64 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            // rowStart vgpr
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            // rowStart vgpr

s_mul_i32 s60, 0x40, s[sgprWorkGroup0]             // s60 = wg0*MT0
_v_add_co_u32 v64, vcc, s60, v64                   // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s62, 0x40, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v65, vcc, s62, v65                   // coord1 = tid1*VW + wg1*MT1


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

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_lshl_u32 v70, v67, v64, 0x4                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[72:73]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[72:73]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[72:73]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[72:73]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[72:73]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[72:73]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[72:73]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[72:73]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[72:73]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[72:73]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[72:73]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[72:73]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[72:73]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[72:73]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[72:73]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[74:75]
v_mul_f64 v[72:73], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[74:75], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[72:73]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[74:75]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[0:3], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[4:7], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[8:11], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[12:15], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[16:19], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[24:27], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[28:31], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[32:35], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[36:39], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[40:43], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[44:47], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[48:51], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[56:59], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[60:63], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_branch label_GW_End_42                           // jump to end
GW_B0_E1_34:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=70 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v70, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v72, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v73, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v74, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v75, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v76, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, -1, v76, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v77, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v78, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v78, -1, v78, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v79, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v80, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v81, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v82, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v83, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v84, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, -1, v84, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v85, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[64:65]               // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[86:87]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[86:87]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[86:87]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[86:87]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[86:87]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[86:87]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[86:87]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[86:87]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[86:87]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[86:87]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[86:87]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[86:87]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[86:87]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[86:87]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[86:87]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[88:89]
v_mul_f64 v[86:87], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[86:87]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[88:89]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[0:3], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[4:7], v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[8:11], v72, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[12:15], v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[16:19], v74, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[24:27], v76, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[28:31], v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[32:35], v78, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[36:39], v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[40:43], v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[44:47], v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[48:51], v82, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[56:59], v84, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[60:63], v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
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

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=17 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v71, v66, v64, 0x4                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64
_buffer_load_b128 v[72:75], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_buffer_load_b128 v[76:79], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_buffer_load_b128 v[80:83], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_buffer_load_b128 v[84:87], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[88:91], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_buffer_load_b128 v[92:95], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_buffer_load_b128 v[96:99], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_buffer_load_b128 v[100:103], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[104:107], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_buffer_load_b128 v[108:111], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_buffer_load_b128 v[112:115], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_buffer_load_b128 v[116:119], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[120:123], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_buffer_load_b128 v[124:127], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_buffer_load_b128 v[128:131], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_buffer_load_b128 v[132:135], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
_v_add_lshl_u32 v70, v67, v64, 0x4                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[136:137]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[136:137]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[136:137]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[136:137]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[136:137]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[136:137]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[136:137]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[136:137]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[136:137]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[136:137]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[136:137]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[136:137]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[136:137]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[136:137]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[136:137]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[136:137]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[138:139]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[72:73], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[74:75], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[72:73], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+2:vgprValuC+2+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[74:75], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+2:vgprValuC+2+1]
_buffer_store_b128 v[0:3], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 1 + 1 - 1
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[76:77], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[78:79], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[76:77], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+6:vgprValuC+6+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[78:79], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+6:vgprValuC+6+1]
_buffer_store_b128 v[4:7], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 2 + 2 - 1
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[80:81], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[82:83], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[80:81], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+10:vgprValuC+10+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[82:83], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+10:vgprValuC+10+1]
_buffer_store_b128 v[8:11], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 3 + 3 - 1
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[84:85], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[86:87], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[84:85], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[86:87], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 4 + 4 - 1
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[88:89], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[90:91], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[88:89], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+18:vgprValuC+18+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[90:91], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+18:vgprValuC+18+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[16:19], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 5 + 5 - 1
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[92:93], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[94:95], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[92:93], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[94:95], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 6 + 6 - 1
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[96:97], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[98:99], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[96:97], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+26:vgprValuC+26+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[98:99], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+26:vgprValuC+26+1]
_buffer_store_b128 v[24:27], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 7 + 7 - 1
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[100:101], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[102:103], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[100:101], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+30:vgprValuC+30+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[102:103], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+30:vgprValuC+30+1]
_buffer_store_b128 v[28:31], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 8 + 8 - 1
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[104:105], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[106:107], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[104:105], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+34:vgprValuC+34+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[106:107], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+34:vgprValuC+34+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[32:35], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 9 + 9 - 1
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[108:109], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[110:111], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[108:109], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+38:vgprValuC+38+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[110:111], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+38:vgprValuC+38+1]
_buffer_store_b128 v[36:39], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 10 + 10 - 1
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[112:113], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[114:115], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[112:113], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+42:vgprValuC+42+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[114:115], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+42:vgprValuC+42+1]
_buffer_store_b128 v[40:43], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 11 + 11 - 1
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[116:117], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[118:119], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[116:117], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+46:vgprValuC+46+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[118:119], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+46:vgprValuC+46+1]
_buffer_store_b128 v[44:47], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 12 + 12 - 1
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[120:121], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[122:123], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[120:121], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+50:vgprValuC+50+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[122:123], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+50:vgprValuC+50+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[48:51], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 13 + 13 - 1
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[124:125], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[126:127], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[124:125], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+54:vgprValuC+54+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[126:127], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+54:vgprValuC+54+1]
_buffer_store_b128 v[52:55], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 14 + 14 - 1
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[128:129], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[130:131], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[128:129], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+58:vgprValuC+58+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[130:131], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+58:vgprValuC+58+1]
_buffer_store_b128 v[56:59], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 15 + 15 - 1
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[132:133], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[134:135], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[132:133], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+62:vgprValuC+62+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[134:135], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+62:vgprValuC+62+1]
_buffer_store_b128 v[60:63], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_branch label_GW_End_42                           // jump to end
GW_B1_E1_41:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=13 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v70, v66, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[72:75], v70, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v70, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[76:79], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v71, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v80, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[84:87], v80, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v80, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v81, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[88:91], v81, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v81, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v82, v66, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[92:95], v82, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v82, v67, v64, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v83, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[96:99], v83, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v83, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v100, v66, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v100, -1, v100, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[104:107], v100, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v100, v67, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v100, -1, v100, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v101, v66, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, -1, v101, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[108:111], v101, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v101, v67, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, -1, v101, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v102, v66, v64, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v102, -1, v102, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[112:115], v102, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v102, v67, v64, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v102, -1, v102, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v103, v66, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, -1, v103, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[116:119], v103, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v103, v67, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, -1, v103, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v120, v66, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v120, -1, v120, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[124:127], v120, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v120, v67, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v120, -1, v120, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v121, v66, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v121, -1, v121, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[128:131], v121, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v121, v67, v68, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v121, -1, v121, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 16                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v66, v66, s60                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v67, v67, s60                           // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v122, v66, v64, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v122, -1, v122, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[132:135], v122, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v122, v67, v64, 0x4                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v122, -1, v122, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0)] */
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[136:137]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[136:137]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[136:137]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[136:137]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[136:137]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[136:137]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[136:137]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[136:137]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[136:137]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[136:137]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[136:137]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[136:137]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[136:137]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[138:139]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[72:73], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[74:75], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[72:73], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+2:vgprValuC+2+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[74:75], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+2:vgprValuC+2+1]
_buffer_store_b128 v[0:3], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[76:77], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[78:79], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[76:77], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+6:vgprValuC+6+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[78:79], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+6:vgprValuC+6+1]
_buffer_store_b128 v[4:7], v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[84:85], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[86:87], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[84:85], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+10:vgprValuC+10+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[86:87], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+10:vgprValuC+10+1]
_buffer_store_b128 v[8:11], v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[88:89], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[90:91], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[88:89], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[90:91], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[92:93], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[94:95], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[92:93], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+18:vgprValuC+18+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[94:95], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+18:vgprValuC+18+1]
_buffer_store_b128 v[16:19], v82, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[96:97], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[98:99], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[96:97], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[98:99], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[104:105], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[106:107], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[104:105], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+26:vgprValuC+26+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[106:107], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+26:vgprValuC+26+1]
_buffer_store_b128 v[24:27], v100, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[108:109], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[110:111], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[108:109], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+30:vgprValuC+30+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[110:111], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+30:vgprValuC+30+1]
_buffer_store_b128 v[28:31], v101, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[112:113], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[114:115], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[112:113], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+34:vgprValuC+34+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[114:115], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+34:vgprValuC+34+1]
_buffer_store_b128 v[32:35], v102, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[116:117], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[118:119], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[116:117], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+38:vgprValuC+38+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[118:119], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+38:vgprValuC+38+1]
_buffer_store_b128 v[36:39], v103, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[124:125], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[126:127], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[124:125], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+42:vgprValuC+42+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[126:127], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+42:vgprValuC+42+1]
_buffer_store_b128 v[40:43], v120, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[128:129], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[130:131], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[128:129], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+46:vgprValuC+46+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[130:131], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+46:vgprValuC+46+1]
_buffer_store_b128 v[44:47], v121, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[132:133], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[134:135], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[132:133], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+50:vgprValuC+50+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[134:135], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+50:vgprValuC+50+1]
_buffer_store_b128 v[48:51], v122, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v68, vcc, v64, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v70, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[72:75], v70, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v70, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v68, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[76:79], v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v71, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_co_u32 v68, vcc, v64, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v80, v66, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[64:65]               // LDC clip if OOB. offset
_buffer_load_b128 v[84:87], v80, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v80, v67, v68, 0x4                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[64:65]               // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f64 v[82:83], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[82:83]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[88:89]
v_mul_f64 v[82:83], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[82:83]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[88:89]
v_mul_f64 v[82:83], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[88:89], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[82:83]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[88:89]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[72:73], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[74:75], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[72:73], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+54:vgprValuC+54+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[74:75], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+54:vgprValuC+54+1]
_buffer_store_b128 v[52:55], v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[76:77], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[78:79], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[76:77], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+58:vgprValuC+58+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[78:79], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+58:vgprValuC+58+1]
_buffer_store_b128 v[56:59], v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[84:85], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[86:87], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[84:85], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+62:vgprValuC+62+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[86:87], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+62:vgprValuC+62+1]
_buffer_store_b128 v[60:63], v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_42                           // jump to end
label_GW_End_42:

label_0047:  /// KernelEnd
s_endpgm                                           // Kernel End

