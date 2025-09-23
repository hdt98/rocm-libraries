
/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureDefault
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1
.globl Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1
.p2align 8
.type Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_user_sgpr_kernarg_preload_length 0
  .amdhsa_user_sgpr_count 2
  .amdhsa_accum_offset 192 // accvgpr offset
  .amdhsa_next_free_vgpr 192 // vgprs
  .amdhsa_next_free_sgpr 74 // sgprs
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
/* ThreadTile= 4 x 8 */
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
  - .name: Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1
    .symbol: 'Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1.kd'
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
    .group_segment_fixed_size:   12544
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 74
    .sgpr_spill_count:           0
    .vgpr_count:                 188
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1:

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
/* ValuC range: [0-128),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 128
.set vgprG2LA, 128
.set vgprValuB_X0_I0, 144
.set vgprG2LB, 144
.set vgprLocalWriteAddrA, 176
.set vgprLocalWriteAddrB, 177
.set vgprGlobalReadOffsetA, 178
.set vgprGlobalReadOffsetB, 179
.set vgprLocalReadAddrA, 180
.set vgprLocalReadAddrB, 181
.set vgprSerial, 182
/* Num VGPR=188 */
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
.set sgprShadowLimitB, 32 // (2)
.set sgprStaggerUIter, 7 // (1)
.set sgprWrapUA, 34 // (2)
.set sgprWrapUB, 64 // (2)
.set sgprGlobalReadIncsA, 66 // (1)
.set sgprGlobalReadIncsB, 67 // (1)
.set sgprScalarGlobalReadOffsetB, 68 // (1)
/* max SGPR=74 */

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
.set constStrideB1J, 1
.set sgprStrideBL, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 64
.set MT1, 128
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
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
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
/* 4x8 thread-tile                        */
/******************************************/
.macro MAC_4x8_X0
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
v_fma_f64 v[vgprValuC+(0+4*4)*4+0:(vgprValuC+0+4*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(0+4*4)*4+0:(vgprValuC+0+4*4)*4+1]
v_fma_f64 v[vgprValuC+(1+4*4)*4+0:(vgprValuC+1+4*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(1+4*4)*4+0:(vgprValuC+1+4*4)*4+1]
v_fma_f64 v[vgprValuC+(2+4*4)*4+0:(vgprValuC+2+4*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(2+4*4)*4+0:(vgprValuC+2+4*4)*4+1]
v_fma_f64 v[vgprValuC+(3+4*4)*4+0:(vgprValuC+3+4*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(3+4*4)*4+0:(vgprValuC+3+4*4)*4+1]
v_fma_f64 v[vgprValuC+(0+5*4)*4+0:(vgprValuC+0+5*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(0+5*4)*4+0:(vgprValuC+0+5*4)*4+1]
v_fma_f64 v[vgprValuC+(1+5*4)*4+0:(vgprValuC+1+5*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(1+5*4)*4+0:(vgprValuC+1+5*4)*4+1]
v_fma_f64 v[vgprValuC+(2+5*4)*4+0:(vgprValuC+2+5*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(2+5*4)*4+0:(vgprValuC+2+5*4)*4+1]
v_fma_f64 v[vgprValuC+(3+5*4)*4+0:(vgprValuC+3+5*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(3+5*4)*4+0:(vgprValuC+3+5*4)*4+1]
v_fma_f64 v[vgprValuC+(0+6*4)*4+0:(vgprValuC+0+6*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(0+6*4)*4+0:(vgprValuC+0+6*4)*4+1]
v_fma_f64 v[vgprValuC+(1+6*4)*4+0:(vgprValuC+1+6*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(1+6*4)*4+0:(vgprValuC+1+6*4)*4+1]
v_fma_f64 v[vgprValuC+(2+6*4)*4+0:(vgprValuC+2+6*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(2+6*4)*4+0:(vgprValuC+2+6*4)*4+1]
v_fma_f64 v[vgprValuC+(3+6*4)*4+0:(vgprValuC+3+6*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(3+6*4)*4+0:(vgprValuC+3+6*4)*4+1]
v_fma_f64 v[vgprValuC+(0+7*4)*4+0:(vgprValuC+0+7*4)*4+1], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(0+7*4)*4+0:(vgprValuC+0+7*4)*4+1]
v_fma_f64 v[vgprValuC+(1+7*4)*4+0:(vgprValuC+1+7*4)*4+1], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(1+7*4)*4+0:(vgprValuC+1+7*4)*4+1]
v_fma_f64 v[vgprValuC+(2+7*4)*4+0:(vgprValuC+2+7*4)*4+1], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(2+7*4)*4+0:(vgprValuC+2+7*4)*4+1]
v_fma_f64 v[vgprValuC+(3+7*4)*4+0:(vgprValuC+3+7*4)*4+1], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(3+7*4)*4+0:(vgprValuC+3+7*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(0+0*4)*4+0:(vgprValuC+0+0*4)*4+1]
v_fma_f64 v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(1+0*4)*4+0:(vgprValuC+1+0*4)*4+1]
v_fma_f64 v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(2+0*4)*4+0:(vgprValuC+2+0*4)*4+1]
v_fma_f64 v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(3+0*4)*4+0:(vgprValuC+3+0*4)*4+1]
v_fma_f64 v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(0+1*4)*4+0:(vgprValuC+0+1*4)*4+1]
v_fma_f64 v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(1+1*4)*4+0:(vgprValuC+1+1*4)*4+1]
v_fma_f64 v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(2+1*4)*4+0:(vgprValuC+2+1*4)*4+1]
v_fma_f64 v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(3+1*4)*4+0:(vgprValuC+3+1*4)*4+1]
v_fma_f64 v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(0+2*4)*4+0:(vgprValuC+0+2*4)*4+1]
v_fma_f64 v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(1+2*4)*4+0:(vgprValuC+1+2*4)*4+1]
v_fma_f64 v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(2+2*4)*4+0:(vgprValuC+2+2*4)*4+1]
v_fma_f64 v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(3+2*4)*4+0:(vgprValuC+3+2*4)*4+1]
v_fma_f64 v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(0+3*4)*4+0:(vgprValuC+0+3*4)*4+1]
v_fma_f64 v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(1+3*4)*4+0:(vgprValuC+1+3*4)*4+1]
v_fma_f64 v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(2+3*4)*4+0:(vgprValuC+2+3*4)*4+1]
v_fma_f64 v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(3+3*4)*4+0:(vgprValuC+3+3*4)*4+1]
v_fma_f64 v[vgprValuC+(0+4*4)*4+0:(vgprValuC+0+4*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(0+4*4)*4+0:(vgprValuC+0+4*4)*4+1]
v_fma_f64 v[vgprValuC+(1+4*4)*4+0:(vgprValuC+1+4*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(1+4*4)*4+0:(vgprValuC+1+4*4)*4+1]
v_fma_f64 v[vgprValuC+(2+4*4)*4+0:(vgprValuC+2+4*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(2+4*4)*4+0:(vgprValuC+2+4*4)*4+1]
v_fma_f64 v[vgprValuC+(3+4*4)*4+0:(vgprValuC+3+4*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(3+4*4)*4+0:(vgprValuC+3+4*4)*4+1]
v_fma_f64 v[vgprValuC+(0+5*4)*4+0:(vgprValuC+0+5*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(0+5*4)*4+0:(vgprValuC+0+5*4)*4+1]
v_fma_f64 v[vgprValuC+(1+5*4)*4+0:(vgprValuC+1+5*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(1+5*4)*4+0:(vgprValuC+1+5*4)*4+1]
v_fma_f64 v[vgprValuC+(2+5*4)*4+0:(vgprValuC+2+5*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(2+5*4)*4+0:(vgprValuC+2+5*4)*4+1]
v_fma_f64 v[vgprValuC+(3+5*4)*4+0:(vgprValuC+3+5*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(3+5*4)*4+0:(vgprValuC+3+5*4)*4+1]
v_fma_f64 v[vgprValuC+(0+6*4)*4+0:(vgprValuC+0+6*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(0+6*4)*4+0:(vgprValuC+0+6*4)*4+1]
v_fma_f64 v[vgprValuC+(1+6*4)*4+0:(vgprValuC+1+6*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(1+6*4)*4+0:(vgprValuC+1+6*4)*4+1]
v_fma_f64 v[vgprValuC+(2+6*4)*4+0:(vgprValuC+2+6*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(2+6*4)*4+0:(vgprValuC+2+6*4)*4+1]
v_fma_f64 v[vgprValuC+(3+6*4)*4+0:(vgprValuC+3+6*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(3+6*4)*4+0:(vgprValuC+3+6*4)*4+1]
v_fma_f64 v[vgprValuC+(0+7*4)*4+0:(vgprValuC+0+7*4)*4+1], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(0+7*4)*4+0:(vgprValuC+0+7*4)*4+1]
v_fma_f64 v[vgprValuC+(1+7*4)*4+0:(vgprValuC+1+7*4)*4+1], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(1+7*4)*4+0:(vgprValuC+1+7*4)*4+1]
v_fma_f64 v[vgprValuC+(2+7*4)*4+0:(vgprValuC+2+7*4)*4+1], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(2+7*4)*4+0:(vgprValuC+2+7*4)*4+1]
v_fma_f64 v[vgprValuC+(3+7*4)*4+0:(vgprValuC+3+7*4)*4+1], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(3+7*4)*4+0:(vgprValuC+3+7*4)*4+1]
v_fma_f64 v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(0+0*4)*4+2:(vgprValuC+0+0*4)*4+3]
v_fma_f64 v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(1+0*4)*4+2:(vgprValuC+1+0*4)*4+3]
v_fma_f64 v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(2+0*4)*4+2:(vgprValuC+2+0*4)*4+3]
v_fma_f64 v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+0*4+2:vgprValuB_X0_I0+0*4+3], v[vgprValuC+(3+0*4)*4+2:(vgprValuC+3+0*4)*4+3]
v_fma_f64 v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(0+1*4)*4+2:(vgprValuC+0+1*4)*4+3]
v_fma_f64 v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(1+1*4)*4+2:(vgprValuC+1+1*4)*4+3]
v_fma_f64 v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(2+1*4)*4+2:(vgprValuC+2+1*4)*4+3]
v_fma_f64 v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+1*4+2:vgprValuB_X0_I0+1*4+3], v[vgprValuC+(3+1*4)*4+2:(vgprValuC+3+1*4)*4+3]
v_fma_f64 v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(0+2*4)*4+2:(vgprValuC+0+2*4)*4+3]
v_fma_f64 v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(1+2*4)*4+2:(vgprValuC+1+2*4)*4+3]
v_fma_f64 v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(2+2*4)*4+2:(vgprValuC+2+2*4)*4+3]
v_fma_f64 v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+2*4+2:vgprValuB_X0_I0+2*4+3], v[vgprValuC+(3+2*4)*4+2:(vgprValuC+3+2*4)*4+3]
v_fma_f64 v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(0+3*4)*4+2:(vgprValuC+0+3*4)*4+3]
v_fma_f64 v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(1+3*4)*4+2:(vgprValuC+1+3*4)*4+3]
v_fma_f64 v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(2+3*4)*4+2:(vgprValuC+2+3*4)*4+3]
v_fma_f64 v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+3*4+2:vgprValuB_X0_I0+3*4+3], v[vgprValuC+(3+3*4)*4+2:(vgprValuC+3+3*4)*4+3]
v_fma_f64 v[vgprValuC+(0+4*4)*4+2:(vgprValuC+0+4*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(0+4*4)*4+2:(vgprValuC+0+4*4)*4+3]
v_fma_f64 v[vgprValuC+(1+4*4)*4+2:(vgprValuC+1+4*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(1+4*4)*4+2:(vgprValuC+1+4*4)*4+3]
v_fma_f64 v[vgprValuC+(2+4*4)*4+2:(vgprValuC+2+4*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(2+4*4)*4+2:(vgprValuC+2+4*4)*4+3]
v_fma_f64 v[vgprValuC+(3+4*4)*4+2:(vgprValuC+3+4*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+4*4+2:vgprValuB_X0_I0+4*4+3], v[vgprValuC+(3+4*4)*4+2:(vgprValuC+3+4*4)*4+3]
v_fma_f64 v[vgprValuC+(0+5*4)*4+2:(vgprValuC+0+5*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(0+5*4)*4+2:(vgprValuC+0+5*4)*4+3]
v_fma_f64 v[vgprValuC+(1+5*4)*4+2:(vgprValuC+1+5*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(1+5*4)*4+2:(vgprValuC+1+5*4)*4+3]
v_fma_f64 v[vgprValuC+(2+5*4)*4+2:(vgprValuC+2+5*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(2+5*4)*4+2:(vgprValuC+2+5*4)*4+3]
v_fma_f64 v[vgprValuC+(3+5*4)*4+2:(vgprValuC+3+5*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+5*4+2:vgprValuB_X0_I0+5*4+3], v[vgprValuC+(3+5*4)*4+2:(vgprValuC+3+5*4)*4+3]
v_fma_f64 v[vgprValuC+(0+6*4)*4+2:(vgprValuC+0+6*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(0+6*4)*4+2:(vgprValuC+0+6*4)*4+3]
v_fma_f64 v[vgprValuC+(1+6*4)*4+2:(vgprValuC+1+6*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(1+6*4)*4+2:(vgprValuC+1+6*4)*4+3]
v_fma_f64 v[vgprValuC+(2+6*4)*4+2:(vgprValuC+2+6*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(2+6*4)*4+2:(vgprValuC+2+6*4)*4+3]
v_fma_f64 v[vgprValuC+(3+6*4)*4+2:(vgprValuC+3+6*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+6*4+2:vgprValuB_X0_I0+6*4+3], v[vgprValuC+(3+6*4)*4+2:(vgprValuC+3+6*4)*4+3]
v_fma_f64 v[vgprValuC+(0+7*4)*4+2:(vgprValuC+0+7*4)*4+3], v[vgprValuA_X0_I0+0*4+0:vgprValuA_X0_I0+0*4+1], -v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(0+7*4)*4+2:(vgprValuC+0+7*4)*4+3]
v_fma_f64 v[vgprValuC+(1+7*4)*4+2:(vgprValuC+1+7*4)*4+3], v[vgprValuA_X0_I0+1*4+0:vgprValuA_X0_I0+1*4+1], -v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(1+7*4)*4+2:(vgprValuC+1+7*4)*4+3]
v_fma_f64 v[vgprValuC+(2+7*4)*4+2:(vgprValuC+2+7*4)*4+3], v[vgprValuA_X0_I0+2*4+0:vgprValuA_X0_I0+2*4+1], -v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(2+7*4)*4+2:(vgprValuC+2+7*4)*4+3]
v_fma_f64 v[vgprValuC+(3+7*4)*4+2:(vgprValuC+3+7*4)*4+3], v[vgprValuA_X0_I0+3*4+0:vgprValuA_X0_I0+3*4+1], -v[vgprValuB_X0_I0+7*4+2:vgprValuB_X0_I0+7*4+3], v[vgprValuC+(3+7*4)*4+2:(vgprValuC+3+7*4)*4+3]
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
v_fma_f64 v[vgprValuC+(0+4*4)*4+2:(vgprValuC+0+4*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(0+4*4)*4+2:(vgprValuC+0+4*4)*4+3]
v_fma_f64 v[vgprValuC+(1+4*4)*4+2:(vgprValuC+1+4*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(1+4*4)*4+2:(vgprValuC+1+4*4)*4+3]
v_fma_f64 v[vgprValuC+(2+4*4)*4+2:(vgprValuC+2+4*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(2+4*4)*4+2:(vgprValuC+2+4*4)*4+3]
v_fma_f64 v[vgprValuC+(3+4*4)*4+2:(vgprValuC+3+4*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+4*4+0:vgprValuB_X0_I0+4*4+1], v[vgprValuC+(3+4*4)*4+2:(vgprValuC+3+4*4)*4+3]
v_fma_f64 v[vgprValuC+(0+5*4)*4+2:(vgprValuC+0+5*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(0+5*4)*4+2:(vgprValuC+0+5*4)*4+3]
v_fma_f64 v[vgprValuC+(1+5*4)*4+2:(vgprValuC+1+5*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(1+5*4)*4+2:(vgprValuC+1+5*4)*4+3]
v_fma_f64 v[vgprValuC+(2+5*4)*4+2:(vgprValuC+2+5*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(2+5*4)*4+2:(vgprValuC+2+5*4)*4+3]
v_fma_f64 v[vgprValuC+(3+5*4)*4+2:(vgprValuC+3+5*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+5*4+0:vgprValuB_X0_I0+5*4+1], v[vgprValuC+(3+5*4)*4+2:(vgprValuC+3+5*4)*4+3]
v_fma_f64 v[vgprValuC+(0+6*4)*4+2:(vgprValuC+0+6*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(0+6*4)*4+2:(vgprValuC+0+6*4)*4+3]
v_fma_f64 v[vgprValuC+(1+6*4)*4+2:(vgprValuC+1+6*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(1+6*4)*4+2:(vgprValuC+1+6*4)*4+3]
v_fma_f64 v[vgprValuC+(2+6*4)*4+2:(vgprValuC+2+6*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(2+6*4)*4+2:(vgprValuC+2+6*4)*4+3]
v_fma_f64 v[vgprValuC+(3+6*4)*4+2:(vgprValuC+3+6*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+6*4+0:vgprValuB_X0_I0+6*4+1], v[vgprValuC+(3+6*4)*4+2:(vgprValuC+3+6*4)*4+3]
v_fma_f64 v[vgprValuC+(0+7*4)*4+2:(vgprValuC+0+7*4)*4+3], v[vgprValuA_X0_I0+0*4+2:vgprValuA_X0_I0+0*4+3], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(0+7*4)*4+2:(vgprValuC+0+7*4)*4+3]
v_fma_f64 v[vgprValuC+(1+7*4)*4+2:(vgprValuC+1+7*4)*4+3], v[vgprValuA_X0_I0+1*4+2:vgprValuA_X0_I0+1*4+3], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(1+7*4)*4+2:(vgprValuC+1+7*4)*4+3]
v_fma_f64 v[vgprValuC+(2+7*4)*4+2:(vgprValuC+2+7*4)*4+3], v[vgprValuA_X0_I0+2*4+2:vgprValuA_X0_I0+2*4+3], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(2+7*4)*4+2:(vgprValuC+2+7*4)*4+3]
v_fma_f64 v[vgprValuC+(3+7*4)*4+2:(vgprValuC+3+7*4)*4+3], v[vgprValuA_X0_I0+3*4+2:vgprValuA_X0_I0+3*4+3], v[vgprValuB_X0_I0+7*4+0:vgprValuB_X0_I0+7*4+1], v[vgprValuC+(3+7*4)*4+2:(vgprValuC+3+7*4)*4+3]
s_setprio 0 // Reset priority after macs
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

Cijk_Alik_BjlkC_ZB_MT64x128x4_SN_K1_TT4_8_WG16_16_1_preloaded: // Kernel start when preloading

/* Load Kernel Args */
_s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 // 
_s_load_b512 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
_s_load_b128 s[56:59], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // 
_s_load_b64 s[60:61], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // 
s_mov_b32 m0, 0x3100                               // LDS clamp at 12544 bytes
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
s_mov_b32 s7, 0x41                                 // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_mul_lo_u32 v0, s7, v0                            // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
_v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x4 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
v_lshlrev_b32 v0, 0x7, v0                          // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x4 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x1100, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)


/* global read addresses: tile offset assignment a */

/* LVCA = 4 */
/* v0 = (local)groA-tile = serial/LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial%LVCA */
v_lshrrev_b32 v0, 2, v[vgprSerial]                 // v0 = v[vgprSerial] / 4
v_and_b32 v1, 3, v[vgprSerial]                     // v1 = v[vgprSerial] % 4
/* gro-unroll *= glvw */
                                                   // v1 = v1 * 1 (multiplier is 1, do nothing)


/* global read addresses: tile offset assignment b */

/* LVCB = 128 */
/* v2 = (local)groB-tile = serial%LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial/LVCB */
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // v3 = v[vgprSerial] / 128
v_and_b32 v2, 127, v[vgprSerial]                   // v2 = v[vgprSerial] % 128
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

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x41, v1     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x4 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v3     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x4 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x1100, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=272*16







s_waitcnt lgkmcnt(0)                               // wait for 152 bytes of kern args
s_sub_u32 s[sgprSrdA+0], s[sgprAddressA+0], 16     // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprAddressA+1], 0     // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprSrdB+0], s[sgprAddressB+0], 16     // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprAddressB+1], 0     // pre-pad to make room for possible pointer shift

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
s_mov_b32 s73, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s71, s[sgprWorkGroup1], s73           // s_magic mul
s_mul_i32 s70, s[sgprWorkGroup1], s73              // s_magic mul
s_lshr_b64 s[70:71], s[70:71], 31                  // sMagicDiv
s_mul_i32 s71, s70, 8                              // quotient * non-magic divisor
s_sub_u32 s71, s[sgprWorkGroup1], s71              // WorkGroup1=remainder
s_mul_i32 s71, s71, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s71, s71, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg1
s_cmp_ge_u32 s70, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s73, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s72, s[sgprWgmRemainder1], 8         // 
s_mul_hi_u32 s3, s71, s73                          // s_magic mul
s_mul_i32 s2, s71, s73                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s72 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s71, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s70, s70, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s70 // wg1 += blockId * WGM


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

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 4 // gROA_0_0_0_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  2,  3, 4 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideBL], 2 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x4 // scalar offset *= bytes/element


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s73, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s72, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s73, s72, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s72, s72, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s72 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s73 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32
s_mul_hi_u32 s71, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s70, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s72, s72, s70                            // accum wg term to tilestart
s_addc_u32 s73, s73, s71                           // accum wg term to tilestart
s_lshl_b64 s[72:73], s[72:73], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s73, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s72, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s72 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s73 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x4 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_mul_hi_u32 s71, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s70, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s72, s72, s70                            // accum wg term to tilestart
s_addc_u32 s73, s73, s71                           // accum wg term to tilestart
s_lshl_b64 s[72:73], s[72:73], 0x4                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeA    // incrA (unrollIdx)


/* global read addresses: increments b */

s_mul_i32 s[sgprGlobalReadIncsB+0], DepthU*BpeB, s[sgprStrideBL] // incrB unrollIdx)

/* declare loop num iterations */



/* initC: remove C-tile 0-128 from pool */

/* initC: remove AB-tile 128-176 from pool */
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
v_mov_b64 v[vgprValuC+64:vgprValuC+64+1], 0x0      // initC
v_mov_b64 v[vgprValuC+66:vgprValuC+66+1], 0x0      // initC
v_mov_b64 v[vgprValuC+68:vgprValuC+68+1], 0x0      // initC
v_mov_b64 v[vgprValuC+70:vgprValuC+70+1], 0x0      // initC
v_mov_b64 v[vgprValuC+72:vgprValuC+72+1], 0x0      // initC
v_mov_b64 v[vgprValuC+74:vgprValuC+74+1], 0x0      // initC
v_mov_b64 v[vgprValuC+76:vgprValuC+76+1], 0x0      // initC
v_mov_b64 v[vgprValuC+78:vgprValuC+78+1], 0x0      // initC
v_mov_b64 v[vgprValuC+80:vgprValuC+80+1], 0x0      // initC
v_mov_b64 v[vgprValuC+82:vgprValuC+82+1], 0x0      // initC
v_mov_b64 v[vgprValuC+84:vgprValuC+84+1], 0x0      // initC
v_mov_b64 v[vgprValuC+86:vgprValuC+86+1], 0x0      // initC
v_mov_b64 v[vgprValuC+88:vgprValuC+88+1], 0x0      // initC
v_mov_b64 v[vgprValuC+90:vgprValuC+90+1], 0x0      // initC
v_mov_b64 v[vgprValuC+92:vgprValuC+92+1], 0x0      // initC
v_mov_b64 v[vgprValuC+94:vgprValuC+94+1], 0x0      // initC
v_mov_b64 v[vgprValuC+96:vgprValuC+96+1], 0x0      // initC
v_mov_b64 v[vgprValuC+98:vgprValuC+98+1], 0x0      // initC
v_mov_b64 v[vgprValuC+100:vgprValuC+100+1], 0x0    // initC
v_mov_b64 v[vgprValuC+102:vgprValuC+102+1], 0x0    // initC
v_mov_b64 v[vgprValuC+104:vgprValuC+104+1], 0x0    // initC
v_mov_b64 v[vgprValuC+106:vgprValuC+106+1], 0x0    // initC
v_mov_b64 v[vgprValuC+108:vgprValuC+108+1], 0x0    // initC
v_mov_b64 v[vgprValuC+110:vgprValuC+110+1], 0x0    // initC
v_mov_b64 v[vgprValuC+112:vgprValuC+112+1], 0x0    // initC
v_mov_b64 v[vgprValuC+114:vgprValuC+114+1], 0x0    // initC
v_mov_b64 v[vgprValuC+116:vgprValuC+116+1], 0x0    // initC
v_mov_b64 v[vgprValuC+118:vgprValuC+118+1], 0x0    // initC
v_mov_b64 v[vgprValuC+120:vgprValuC+120+1], 0x0    // initC
v_mov_b64 v[vgprValuC+122:vgprValuC+122+1], 0x0    // initC
v_mov_b64 v[vgprValuC+124:vgprValuC+124+1], 0x0    // initC
v_mov_b64 v[vgprValuC+126:vgprValuC+126+1], 0x0    // initC

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 2 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 4
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

s_and_b32 s[sgprStaggerUIter], s[sgprOrigStaggerUIter], s[sgprWorkGroup0] // Compute actual stagger start for this tile


/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s71, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s70, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32


/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s71, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s70, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 1 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_10:
s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

label_0011: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimitB // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=05wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //PGR=0, prior iter done reading lds


/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4096 // lwoB_0_0_1_0 = (0*LSCB) + (1*LSPB)(*MT1J+PAD) = 4096

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-12prefetch wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* iter 0 */


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
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->65 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x8_X0

/* iter 1 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1040 // L -> Reg lro=65 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1296 // L -> Reg lro=65 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:1552 // L -> Reg lro=65 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:1808 // L -> Reg lro=65 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->130 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x8_X0

/* iter 2 */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:2080 // L -> Reg lro=130 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:2336 // L -> Reg lro=130 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:2592 // L -> Reg lro=130 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:2848 // L -> Reg lro=130 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:4096 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:4352 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:4608 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:4864 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:5376 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:5632 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:5888 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->195 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x8_X0

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
_ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:3120 // L -> Reg lro=195 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:3376 // L -> Reg lro=195 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:3632 // L -> Reg lro=195 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:3888 // L -> Reg lro=195 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
_ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:6144 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:6400 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:6656 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:6912 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:7168 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:7424 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:7936 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x8_X0

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Tail Loop                              */
/******************************************/


//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 3, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 4
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_cbranch_scc1 SkipTailLoopL_8                     // skip to end of tail loop b/c numIter==0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment


/* remove stagger offsets for tail loop */

s_mov_b32 s72, 2                                   // 
s_mul_hi_u32 s71, s72, s[sgprGlobalReadIncsA+0]    // 2 * GlobalReadIncs
s_mul_i32 s70, s72, s[sgprGlobalReadIncsA+0]       // 2 * GlobalReadIncs
s_mul_hi_u32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s70, s70, s72                            // start offset S in bytes
s_subb_u32 s71, s71, s73                           // start offset S in bytes
s_sub_u32 s70, s70, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimitA // Move shadow to real if we are within 2^32

s_mov_b32 s72, 2                                   // 
s_mul_hi_u32 s71, s72, s[sgprGlobalReadIncsB+0]    // 2 * GlobalReadIncs
s_mul_i32 s70, s72, s[sgprGlobalReadIncsB+0]       // 2 * GlobalReadIncs
s_mul_hi_u32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] // StaggerUIter * GlobalReadIncs
s_sub_u32 s70, s70, s72                            // start offset S in bytes
s_subb_u32 s71, s71, s73                           // start offset S in bytes
s_sub_u32 s70, s70, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s71      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
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
/* g2l=4, load component 0 */
_buffer_load_b128 v[vgprG2LB+4+0:vgprG2LB+4+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // load one buffer value

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* Done global A/B reads */




/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4096 // lwoB_0_0_1_0 = (0*LSCB) + (1*LSPB)(*MT1J+PAD) = 4096


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


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
_ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s62, 0x410                               // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s62, v[vgprLocalReadAddrA] // lrA += 1040 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s62, 0x800                               // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s62, v[vgprLocalReadAddrB] // lrB += 2048 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read

MAC_4x8_X0

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_6                    // restart LoopL
TailLoopEndL_7:

SkipTailLoopL_8:

Summation_End_14:
/* endSummation: add vgpr [128...180) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set WrapUB, UNDEF
.set GlobalReadIncsA, UNDEF
.set GlobalReadIncsB, UNDEF
.set ScalarGlobalReadOffsetB, UNDEF

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], BufferOOB                 // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s62, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s61, s62, s[sgprStrideC1J]            // CScale s62 by Stride
s_mul_i32 s60, s62, s[sgprStrideC1J]               // CScale s62 by Stride
s_lshl_b64 s[60:61], s[60:61], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s60    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s61   // add hi to SRD
s_mul_hi_u32 s61, s62, s[sgprStrideD1J]            // Scale s62 by Stride
s_mul_i32 s60, s62, s[sgprStrideD1J]               // Scale s62 by Stride
s_lshl_b64 s[60:61], s[60:61], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s60    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s61   // add hi to SRD

s_mul_hi_u32 s61, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s60, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[60:61], s[60:61], 4                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s60        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s61       // add hi to SRD
s_mul_hi_u32 s61, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s60, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[60:61], s[60:61], 4                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s60        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s61       // add hi to SRD




/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v129, 4, v[vgprSerial]               // v129 = v[vgprSerial] / 16
v_and_b32 v128, 15, v[vgprSerial]                  // v128 = v[vgprSerial] % 16
                                                   // v128 = v128 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr

s_mul_i32 s60, 0x40, s[sgprWorkGroup0]             // s60 = wg0*MT0
_v_add_co_u32 v128, vcc, s60, v128                 // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s62, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v129, vcc, s62, v129                 // coord1 = tid1*VW + wg1*MT1


/* not-LocalSplitU: global write */

s_mov_b32 s59, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s59, s[sgprBeta+1], s59                   // tmp |= Beta[1] 
s_or_b32 s59, s[sgprBeta+2], s59                   // tmp |= Beta[2] 
s_or_b32 s59, s[sgprBeta+3], s59                   // tmp |= Beta[3] 
s_cmpk_eq_u32 s59, 0x0                             // Beta == 0
s_cbranch_scc0 GW_Beta_29                          // Branch if Beta is not zero

s_and_b32 s60, 63, s[sgprSizeI]                    // s60 = s[sgprSizeI] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s61                // wg0 >= nwg0-1 ?
s_cselect_b32 s60, s60, 0                          // set rMT0
s_cmpk_gt_u32 s60, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_20                         // jump if edges required
s_and_b32 s60, 127, s[sgprSizeJ]                   // s60 = s[sgprSizeJ] % 128
s_add_u32 s61, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s61                // wg1 >= nwg1-1
s_cselect_b32 s60, s60, 0                          // set rMT1
s_cmpk_gt_u32 s60, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_20                         // jump if edges required
GW_B0_E0_17:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=32 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
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
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
_v_add_lshl_u32 v134, v131, v128, 0x4              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
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
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[136:137]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[136:137]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[136:137]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+78:vgprValuC+78+1], v[136:137]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+78:vgprValuC+78+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[136:137]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+86:vgprValuC+86+1], v[136:137]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+86:vgprValuC+86+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[136:137]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+94:vgprValuC+94+1], v[136:137]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+94:vgprValuC+94+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[136:137]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+102:vgprValuC+102+1], v[136:137]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+102:vgprValuC+102+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[136:137]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+110:vgprValuC+110+1], v[136:137]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+110:vgprValuC+110+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+114:vgprValuC+114+1], v[136:137]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+114:vgprValuC+114+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+118:vgprValuC+118+1], v[136:137]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+118:vgprValuC+118+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+122:vgprValuC+122+1], v[136:137]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+122:vgprValuC+122+1], v[138:139]
v_mul_f64 v[136:137], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_mul_f64 v[138:139], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+126:vgprValuC+126+1], v[136:137]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+126:vgprValuC+126+1], v[138:139]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[0:3], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[4:7], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[8:11], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[12:15], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[16:19], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[24:27], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[28:31], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[32:35], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[36:39], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[40:43], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[44:47], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[48:51], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[56:59], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[60:63], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[64:67], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[68:71], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[72:75], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[76:79], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[80:83], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[84:87], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[88:91], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[92:95], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[96:99], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[100:103], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[104:107], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[108:111], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[112:115], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[116:119], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
_buffer_store_b128 v[120:123], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
_buffer_store_b128 v[124:127], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_branch label_GW_End_28                           // jump to end
GW_B0_E1_20:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=46 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v137, -1, v137, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v139, -1, v139, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v140, -1, v140, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v141, -1, v141, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v142, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v143, -1, v143, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v146, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v149, -1, v149, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v150, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v151, -1, v151, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, -1, v152, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v153, -1, v153, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v154, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v155, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v155, -1, v155, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v156, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, -1, v156, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v157, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v157, -1, v157, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v158, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v159, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v159, -1, v159, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v160, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v160, -1, v160, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v161, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v161, -1, v161, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v162, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v162, -1, v162, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v163, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v163, -1, v163, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v164, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v165, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[166:167]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[166:167]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[166:167]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[166:167]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[166:167]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[166:167]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[166:167]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[166:167]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[166:167]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[166:167]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[166:167]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[166:167]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[166:167]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[166:167]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[166:167]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[166:167]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[166:167]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[166:167]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[166:167]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+78:vgprValuC+78+1], v[166:167]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+78:vgprValuC+78+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[166:167]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+86:vgprValuC+86+1], v[166:167]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+86:vgprValuC+86+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[166:167]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+94:vgprValuC+94+1], v[166:167]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+94:vgprValuC+94+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[166:167]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+102:vgprValuC+102+1], v[166:167]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+102:vgprValuC+102+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[166:167]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+110:vgprValuC+110+1], v[166:167]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+110:vgprValuC+110+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+114:vgprValuC+114+1], v[166:167]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+114:vgprValuC+114+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+118:vgprValuC+118+1], v[166:167]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+118:vgprValuC+118+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+122:vgprValuC+122+1], v[166:167]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+122:vgprValuC+122+1], v[168:169]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_mul_f64 v[168:169], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+126:vgprValuC+126+1], v[166:167]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+126:vgprValuC+126+1], v[168:169]

/* apply mask, calc new C and issue writes */
_buffer_store_b128 v[0:3], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[4:7], v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[8:11], v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[12:15], v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[16:19], v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[20:23], v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[24:27], v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[28:31], v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[32:35], v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[36:39], v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[40:43], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[44:47], v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[48:51], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[52:55], v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[56:59], v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[60:63], v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[64:67], v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[68:71], v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[72:75], v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[76:79], v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[80:83], v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[84:87], v155, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[88:91], v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[92:95], v157, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[96:99], v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[100:103], v159, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[104:107], v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[108:111], v161, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[112:115], v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[116:119], v163, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[120:123], v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_buffer_store_b128 v[124:127], v165, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_28                           // jump to end
GW_Beta_29:
s_and_b32 s60, 63, s[sgprSizeI]                    // s60 = s[sgprSizeI] % 64
s_add_u32 s61, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s61                // wg0 >= nwg0-1 ?
s_cselect_b32 s60, s60, 0                          // set rMT0
s_cmpk_gt_u32 s60, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_27                         // jump if edges required
s_and_b32 s60, 127, s[sgprSizeJ]                   // s60 = s[sgprSizeJ] % 128
s_add_u32 s61, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s61                // wg1 >= nwg1-1
s_cselect_b32 s60, s60, 0                          // set rMT1
s_cmpk_gt_u32 s60, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_27                         // jump if edges required
GW_B1_E0_24:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=11 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v135, v130, v128, 0x4              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
_buffer_load_b128 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_buffer_load_b128 v[144:147], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_buffer_load_b128 v[148:151], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[152:155], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_buffer_load_b128 v[156:159], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_buffer_load_b128 v[160:163], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_buffer_load_b128 v[164:167], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[168:171], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_buffer_load_b128 v[172:175], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_buffer_load_b128 v[176:179], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x4              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0)] */
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[184:185]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[184:185]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[184:185]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[184:185]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[184:185]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[184:185]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[184:185]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[184:185]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[184:185]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[184:185]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[184:185]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[186:187]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 0 + 0 - 1
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+2:vgprValuC+2+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+2:vgprValuC+2+1]
_buffer_store_b128 v[0:3], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 1 + 1 - 1
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+6:vgprValuC+6+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+6:vgprValuC+6+1]
_buffer_store_b128 v[4:7], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 2 + 2 - 1
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[144:145], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[146:147], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[144:145], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+10:vgprValuC+10+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[146:147], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+10:vgprValuC+10+1]
_buffer_store_b128 v[8:11], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 3 + 3 - 1
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 4 + 4 - 1
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+18:vgprValuC+18+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+18:vgprValuC+18+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[16:19], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 5 + 5 - 1
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 6 + 6 - 1
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+26:vgprValuC+26+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+26:vgprValuC+26+1]
_buffer_store_b128 v[24:27], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 7 + 7 - 1
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[164:165], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[166:167], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[164:165], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+30:vgprValuC+30+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[166:167], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+30:vgprValuC+30+1]
_buffer_store_b128 v[28:31], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 8 + 8 - 1
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+34:vgprValuC+34+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+34:vgprValuC+34+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[32:35], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 9 + 9 - 1
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+38:vgprValuC+38+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+38:vgprValuC+38+1]
_buffer_store_b128 v[36:39], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 10 + 10 - 1
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[176:177], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[178:179], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[176:177], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+42:vgprValuC+42+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[178:179], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+42:vgprValuC+42+1]
_buffer_store_b128 v[40:43], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_buffer_load_b128 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_buffer_load_b128 v[144:147], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_buffer_load_b128 v[148:151], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_buffer_load_b128 v[152:155], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[156:159], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_buffer_load_b128 v[160:163], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_buffer_load_b128 v[164:167], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
_buffer_load_b128 v[168:171], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[172:175], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_buffer_load_b128 v[176:179], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc

/* rC *= alpha batchElements=[(2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0)] */
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[184:185]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[184:185]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[184:185]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[184:185]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[184:185]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[184:185]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[184:185]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[184:185]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+78:vgprValuC+78+1], v[184:185]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+78:vgprValuC+78+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[184:185]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[186:187]
v_mul_f64 v[184:185], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_mul_f64 v[186:187], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+86:vgprValuC+86+1], v[184:185]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+86:vgprValuC+86+1], v[186:187]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 0 + 0 - 1
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+46:vgprValuC+46+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+46:vgprValuC+46+1]
_buffer_store_b128 v[44:47], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 1 + 1 - 1
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+50:vgprValuC+50+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+50:vgprValuC+50+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[48:51], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 2 + 2 - 1
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[144:145], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[146:147], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[144:145], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+54:vgprValuC+54+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[146:147], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+54:vgprValuC+54+1]
_buffer_store_b128 v[52:55], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 3 + 3 - 1
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+58:vgprValuC+58+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+58:vgprValuC+58+1]
_buffer_store_b128 v[56:59], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 4 + 4 - 1
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+62:vgprValuC+62+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+62:vgprValuC+62+1]
_buffer_store_b128 v[60:63], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 5 + 5 - 1
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+66:vgprValuC+66+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+66:vgprValuC+66+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[64:67], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 6 + 6 - 1
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+70:vgprValuC+70+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+70:vgprValuC+70+1]
_buffer_store_b128 v[68:71], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 7 + 7 - 1
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[164:165], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[166:167], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[164:165], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+74:vgprValuC+74+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[166:167], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+74:vgprValuC+74+1]
_buffer_store_b128 v[72:75], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 8 + 8 - 1
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+76:vgprValuC+76+1]
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+76:vgprValuC+76+1]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+78:vgprValuC+78+1]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+78:vgprValuC+78+1]
_buffer_store_b128 v[76:79], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 9 + 9 - 1
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+82:vgprValuC+82+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+82:vgprValuC+82+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[80:83], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 11 - 10 + 10 - 1
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], v[176:177], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+84:vgprValuC+84+1]
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], v[178:179], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+84:vgprValuC+84+1]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], v[176:177], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+86:vgprValuC+86+1]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], v[178:179], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+86:vgprValuC+86+1]
_buffer_store_b128 v[84:87], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Alpha Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_buffer_load_b128 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[144:147], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_buffer_load_b128 v[148:151], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_buffer_load_b128 v[152:155], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
_buffer_load_b128 v[156:159], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s60, s[sgprStrideC1J], 256               // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_load_b128 v[160:163], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_buffer_load_b128 v[164:167], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:256 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_buffer_load_b128 v[168:171], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:512 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
_buffer_load_b128 v[172:175], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:768 // load C for beta calc

/* rC *= alpha batchElements=[(5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[176:177]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+94:vgprValuC+94+1], v[176:177]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+94:vgprValuC+94+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[176:177]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+102:vgprValuC+102+1], v[176:177]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+102:vgprValuC+102+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[176:177]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+110:vgprValuC+110+1], v[176:177]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+110:vgprValuC+110+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+114:vgprValuC+114+1], v[176:177]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+114:vgprValuC+114+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+118:vgprValuC+118+1], v[176:177]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+118:vgprValuC+118+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+122:vgprValuC+122+1], v[176:177]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+122:vgprValuC+122+1], v[178:179]
v_mul_f64 v[176:177], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_mul_f64 v[178:179], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+126:vgprValuC+126+1], v[176:177]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+126:vgprValuC+126+1], v[178:179]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 0 + 0 - 1
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+90:vgprValuC+90+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+90:vgprValuC+90+1]
_buffer_store_b128 v[88:91], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 1 + 1 - 1
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+92:vgprValuC+92+1]
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+92:vgprValuC+92+1]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+94:vgprValuC+94+1]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+94:vgprValuC+94+1]
_buffer_store_b128 v[92:95], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 2 + 2 - 1
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[144:145], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[146:147], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[144:145], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+98:vgprValuC+98+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[146:147], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+98:vgprValuC+98+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[96:99], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 3 + 3 - 1
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+100:vgprValuC+100+1]
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+100:vgprValuC+100+1]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+102:vgprValuC+102+1]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+102:vgprValuC+102+1]
_buffer_store_b128 v[100:103], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 4 + 4 - 1
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+106:vgprValuC+106+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+106:vgprValuC+106+1]
_buffer_store_b128 v[104:107], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 5 + 5 - 1
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+108:vgprValuC+108+1]
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+108:vgprValuC+108+1]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+110:vgprValuC+110+1]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+110:vgprValuC+110+1]
_buffer_store_b128 v[108:111], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 6 + 6 - 1
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+114:vgprValuC+114+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+114:vgprValuC+114+1]
s_mul_i32 s60, s[sgprStrideD1J], 256               // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s60       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
_buffer_store_b128 v[112:115], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 7 + 7 - 1
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[164:165], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[166:167], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[164:165], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+118:vgprValuC+118+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[166:167], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+118:vgprValuC+118+1]
_buffer_store_b128 v[116:119], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:256 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 8 + 8 - 1
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+122:vgprValuC+122+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+122:vgprValuC+122+1]
_buffer_store_b128 v[120:123], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:512 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 9 + 9 - 1
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+126:vgprValuC+126+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+126:vgprValuC+126+1]
_buffer_store_b128 v[124:127], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:768 // store D
s_branch label_GW_End_28                           // jump to end
GW_B1_E1_27:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=8 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[136:139], v134, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v135, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[148:151], v144, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v145, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[152:155], v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[156:159], v146, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v147, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[160:163], v147, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v147, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[168:171], v164, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v165, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[172:175], v165, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v165, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0)] */
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+0:vgprValuC+0+1] // 
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+2:vgprValuC+2+1], v[166:167]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+2:vgprValuC+2+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+4:vgprValuC+4+1] // 
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+6:vgprValuC+6+1], v[166:167]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+6:vgprValuC+6+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+8:vgprValuC+8+1] // 
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+10:vgprValuC+10+1], v[166:167]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+10:vgprValuC+10+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+12:vgprValuC+12+1] // 
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+14:vgprValuC+14+1], v[166:167]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+14:vgprValuC+14+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+16:vgprValuC+16+1] // 
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+18:vgprValuC+18+1], v[166:167]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+18:vgprValuC+18+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+20:vgprValuC+20+1] // 
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+22:vgprValuC+22+1], v[166:167]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+22:vgprValuC+22+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+24:vgprValuC+24+1] // 
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+26:vgprValuC+26+1], v[166:167]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+26:vgprValuC+26+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+28:vgprValuC+28+1] // 
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+30:vgprValuC+30+1], v[166:167]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+30:vgprValuC+30+1], v[176:177]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+0:vgprValuC+0+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+0:vgprValuC+0+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+2:vgprValuC+2+1]
v_fma_f64 v[vgprValuC+2:vgprValuC+2+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+2:vgprValuC+2+1]
_buffer_store_b128 v[0:3], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+4:vgprValuC+4+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+4:vgprValuC+4+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+6:vgprValuC+6+1]
v_fma_f64 v[vgprValuC+6:vgprValuC+6+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+6:vgprValuC+6+1]
_buffer_store_b128 v[4:7], v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+8:vgprValuC+8+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+8:vgprValuC+8+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+10:vgprValuC+10+1]
v_fma_f64 v[vgprValuC+10:vgprValuC+10+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+10:vgprValuC+10+1]
_buffer_store_b128 v[8:11], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+12:vgprValuC+12+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+12:vgprValuC+12+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+14:vgprValuC+14+1]
v_fma_f64 v[vgprValuC+14:vgprValuC+14+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+14:vgprValuC+14+1]
_buffer_store_b128 v[12:15], v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+16:vgprValuC+16+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+16:vgprValuC+16+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+18:vgprValuC+18+1]
v_fma_f64 v[vgprValuC+18:vgprValuC+18+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+18:vgprValuC+18+1]
_buffer_store_b128 v[16:19], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+20:vgprValuC+20+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+20:vgprValuC+20+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+22:vgprValuC+22+1]
v_fma_f64 v[vgprValuC+22:vgprValuC+22+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+22:vgprValuC+22+1]
_buffer_store_b128 v[20:23], v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+24:vgprValuC+24+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+24:vgprValuC+24+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+26:vgprValuC+26+1]
v_fma_f64 v[vgprValuC+26:vgprValuC+26+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+26:vgprValuC+26+1]
_buffer_store_b128 v[24:27], v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+28:vgprValuC+28+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+28:vgprValuC+28+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+30:vgprValuC+30+1]
v_fma_f64 v[vgprValuC+30:vgprValuC+30+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+30:vgprValuC+30+1]
_buffer_store_b128 v[28:31], v165, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[136:139], v134, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v135, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[148:151], v144, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v145, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[152:155], v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[156:159], v146, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v147, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[160:163], v147, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v147, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[168:171], v164, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v165, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[172:175], v165, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v165, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+32:vgprValuC+32+1] // 
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+34:vgprValuC+34+1], v[166:167]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+34:vgprValuC+34+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+36:vgprValuC+36+1] // 
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+38:vgprValuC+38+1], v[166:167]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+38:vgprValuC+38+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+40:vgprValuC+40+1] // 
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+42:vgprValuC+42+1], v[166:167]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+42:vgprValuC+42+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+44:vgprValuC+44+1] // 
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+46:vgprValuC+46+1], v[166:167]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+46:vgprValuC+46+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+48:vgprValuC+48+1] // 
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+50:vgprValuC+50+1], v[166:167]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+50:vgprValuC+50+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+52:vgprValuC+52+1] // 
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+54:vgprValuC+54+1], v[166:167]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+54:vgprValuC+54+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+56:vgprValuC+56+1] // 
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+58:vgprValuC+58+1], v[166:167]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+58:vgprValuC+58+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+60:vgprValuC+60+1] // 
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+62:vgprValuC+62+1], v[166:167]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+62:vgprValuC+62+1], v[176:177]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+32:vgprValuC+32+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+32:vgprValuC+32+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+34:vgprValuC+34+1]
v_fma_f64 v[vgprValuC+34:vgprValuC+34+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+34:vgprValuC+34+1]
_buffer_store_b128 v[32:35], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+36:vgprValuC+36+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+36:vgprValuC+36+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+38:vgprValuC+38+1]
v_fma_f64 v[vgprValuC+38:vgprValuC+38+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+38:vgprValuC+38+1]
_buffer_store_b128 v[36:39], v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+40:vgprValuC+40+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+40:vgprValuC+40+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+42:vgprValuC+42+1]
v_fma_f64 v[vgprValuC+42:vgprValuC+42+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+42:vgprValuC+42+1]
_buffer_store_b128 v[40:43], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+44:vgprValuC+44+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+44:vgprValuC+44+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+46:vgprValuC+46+1]
v_fma_f64 v[vgprValuC+46:vgprValuC+46+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+46:vgprValuC+46+1]
_buffer_store_b128 v[44:47], v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+48:vgprValuC+48+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+48:vgprValuC+48+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+50:vgprValuC+50+1]
v_fma_f64 v[vgprValuC+50:vgprValuC+50+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+50:vgprValuC+50+1]
_buffer_store_b128 v[48:51], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+52:vgprValuC+52+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+52:vgprValuC+52+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+54:vgprValuC+54+1]
v_fma_f64 v[vgprValuC+54:vgprValuC+54+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+54:vgprValuC+54+1]
_buffer_store_b128 v[52:55], v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+56:vgprValuC+56+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+56:vgprValuC+56+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+58:vgprValuC+58+1]
v_fma_f64 v[vgprValuC+58:vgprValuC+58+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+58:vgprValuC+58+1]
_buffer_store_b128 v[56:59], v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+60:vgprValuC+60+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+60:vgprValuC+60+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+62:vgprValuC+62+1]
v_fma_f64 v[vgprValuC+62:vgprValuC+62+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+62:vgprValuC+62+1]
_buffer_store_b128 v[60:63], v165, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[136:139], v134, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v135, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[148:151], v144, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v145, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[152:155], v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[156:159], v146, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v147, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[160:163], v147, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v147, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[168:171], v164, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v165, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[172:175], v165, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v165, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0)] */
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+64:vgprValuC+64+1] // 
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+66:vgprValuC+66+1], v[166:167]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+66:vgprValuC+66+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+68:vgprValuC+68+1] // 
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+70:vgprValuC+70+1], v[166:167]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+70:vgprValuC+70+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+72:vgprValuC+72+1] // 
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+74:vgprValuC+74+1], v[166:167]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+74:vgprValuC+74+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+76:vgprValuC+76+1] // 
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+78:vgprValuC+78+1], v[166:167]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+78:vgprValuC+78+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+80:vgprValuC+80+1] // 
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+82:vgprValuC+82+1], v[166:167]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+82:vgprValuC+82+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+84:vgprValuC+84+1] // 
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+86:vgprValuC+86+1], v[166:167]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+86:vgprValuC+86+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+88:vgprValuC+88+1] // 
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+90:vgprValuC+90+1], v[166:167]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+90:vgprValuC+90+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+92:vgprValuC+92+1] // 
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+94:vgprValuC+94+1], v[166:167]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+94:vgprValuC+94+1], v[176:177]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+64:vgprValuC+64+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+64:vgprValuC+64+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+66:vgprValuC+66+1]
v_fma_f64 v[vgprValuC+66:vgprValuC+66+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+66:vgprValuC+66+1]
_buffer_store_b128 v[64:67], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+68:vgprValuC+68+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+68:vgprValuC+68+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+70:vgprValuC+70+1]
v_fma_f64 v[vgprValuC+70:vgprValuC+70+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+70:vgprValuC+70+1]
_buffer_store_b128 v[68:71], v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+72:vgprValuC+72+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+72:vgprValuC+72+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+74:vgprValuC+74+1]
v_fma_f64 v[vgprValuC+74:vgprValuC+74+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+74:vgprValuC+74+1]
_buffer_store_b128 v[72:75], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+76:vgprValuC+76+1]
v_fma_f64 v[vgprValuC+76:vgprValuC+76+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+76:vgprValuC+76+1]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+78:vgprValuC+78+1]
v_fma_f64 v[vgprValuC+78:vgprValuC+78+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+78:vgprValuC+78+1]
_buffer_store_b128 v[76:79], v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+80:vgprValuC+80+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+80:vgprValuC+80+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+82:vgprValuC+82+1]
v_fma_f64 v[vgprValuC+82:vgprValuC+82+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+82:vgprValuC+82+1]
_buffer_store_b128 v[80:83], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+84:vgprValuC+84+1]
v_fma_f64 v[vgprValuC+84:vgprValuC+84+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+84:vgprValuC+84+1]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+86:vgprValuC+86+1]
v_fma_f64 v[vgprValuC+86:vgprValuC+86+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+86:vgprValuC+86+1]
_buffer_store_b128 v[84:87], v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+88:vgprValuC+88+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+88:vgprValuC+88+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+90:vgprValuC+90+1]
v_fma_f64 v[vgprValuC+90:vgprValuC+90+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+90:vgprValuC+90+1]
_buffer_store_b128 v[88:91], v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+92:vgprValuC+92+1]
v_fma_f64 v[vgprValuC+92:vgprValuC+92+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+92:vgprValuC+92+1]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+94:vgprValuC+94+1]
v_fma_f64 v[vgprValuC+94:vgprValuC+94+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+94:vgprValuC+94+1]
_buffer_store_b128 v[92:95], v165, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Alpha Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[136:139], v134, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[140:143], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v135, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[148:151], v144, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v145, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[152:155], v145, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v145, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s60, s[sgprStrideC1J], 16                // scale stride
_v_add_u32 v130, v130, s60                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s60, s[sgprStrideD1J], 16                // scale stride
_v_add_u32 v131, v131, s60                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[60:61], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[156:159], v146, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v147, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[160:163], v147, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v147, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, -1, v147, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[168:171], v164, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v164, -1, v164, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[60:61], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[60:61], s[64:65]             // in0 && in1
_v_add_lshl_u32 v165, v130, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDC clip if OOB. offset
_buffer_load_b128 v[172:175], v165, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
_v_add_lshl_u32 v165, v131, v132, 0x4              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v165, -1, v165, s[64:65]             // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+96:vgprValuC+96+1] // 
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+98:vgprValuC+98+1], v[166:167]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+98:vgprValuC+98+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+100:vgprValuC+100+1] // 
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+102:vgprValuC+102+1], v[166:167]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+102:vgprValuC+102+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+104:vgprValuC+104+1] // 
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+106:vgprValuC+106+1], v[166:167]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+106:vgprValuC+106+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+108:vgprValuC+108+1] // 
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+110:vgprValuC+110+1], v[166:167]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+110:vgprValuC+110+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+112:vgprValuC+112+1] // 
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+114:vgprValuC+114+1], v[166:167]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+114:vgprValuC+114+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+116:vgprValuC+116+1] // 
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+118:vgprValuC+118+1], v[166:167]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+118:vgprValuC+118+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+120:vgprValuC+120+1] // 
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+122:vgprValuC+122+1], v[166:167]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+122:vgprValuC+122+1], v[176:177]
v_mul_f64 v[166:167], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_mul_f64 v[176:177], s[sgprAlpha+2:sgprAlpha+2+1], v[vgprValuC+124:vgprValuC+124+1] // 
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], s[sgprAlpha+2:sgprAlpha+2+1], -v[vgprValuC+126:vgprValuC+126+1], v[166:167]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], s[sgprAlpha+0:sgprAlpha+0+1], v[vgprValuC+126:vgprValuC+126+1], v[176:177]
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[136:137], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+96:vgprValuC+96+1], v[138:139], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+96:vgprValuC+96+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[136:137], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+98:vgprValuC+98+1]
v_fma_f64 v[vgprValuC+98:vgprValuC+98+1], v[138:139], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+98:vgprValuC+98+1]
_buffer_store_b128 v[96:99], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], v[140:141], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+100:vgprValuC+100+1]
v_fma_f64 v[vgprValuC+100:vgprValuC+100+1], v[142:143], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+100:vgprValuC+100+1]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], v[140:141], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+102:vgprValuC+102+1]
v_fma_f64 v[vgprValuC+102:vgprValuC+102+1], v[142:143], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+102:vgprValuC+102+1]
_buffer_store_b128 v[100:103], v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[148:149], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+104:vgprValuC+104+1], v[150:151], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+104:vgprValuC+104+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[148:149], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+106:vgprValuC+106+1]
v_fma_f64 v[vgprValuC+106:vgprValuC+106+1], v[150:151], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+106:vgprValuC+106+1]
_buffer_store_b128 v[104:107], v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], v[152:153], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+108:vgprValuC+108+1]
v_fma_f64 v[vgprValuC+108:vgprValuC+108+1], v[154:155], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+108:vgprValuC+108+1]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], v[152:153], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+110:vgprValuC+110+1]
v_fma_f64 v[vgprValuC+110:vgprValuC+110+1], v[154:155], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+110:vgprValuC+110+1]
_buffer_store_b128 v[108:111], v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[156:157], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+112:vgprValuC+112+1], v[158:159], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+112:vgprValuC+112+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[156:157], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+114:vgprValuC+114+1]
v_fma_f64 v[vgprValuC+114:vgprValuC+114+1], v[158:159], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+114:vgprValuC+114+1]
_buffer_store_b128 v[112:115], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[160:161], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+116:vgprValuC+116+1], v[162:163], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+116:vgprValuC+116+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[160:161], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+118:vgprValuC+118+1]
v_fma_f64 v[vgprValuC+118:vgprValuC+118+1], v[162:163], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+118:vgprValuC+118+1]
_buffer_store_b128 v[116:119], v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[168:169], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+120:vgprValuC+120+1], v[170:171], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+120:vgprValuC+120+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[168:169], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+122:vgprValuC+122+1]
v_fma_f64 v[vgprValuC+122:vgprValuC+122+1], v[170:171], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+122:vgprValuC+122+1]
_buffer_store_b128 v[120:123], v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[172:173], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+124:vgprValuC+124+1], v[174:175], -s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+124:vgprValuC+124+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[172:173], s[sgprBeta+2:sgprBeta+2+1], v[vgprValuC+126:vgprValuC+126+1]
v_fma_f64 v[vgprValuC+126:vgprValuC+126+1], v[174:175], s[sgprBeta+0:sgprBeta+0+1], v[vgprValuC+126:vgprValuC+126+1]
_buffer_store_b128 v[124:127], v165, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_28                           // jump to end
label_GW_End_28:

label_0033:  /// KernelEnd
s_endpgm                                           // Kernel End

