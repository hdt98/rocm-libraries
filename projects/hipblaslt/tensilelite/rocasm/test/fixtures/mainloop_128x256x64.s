label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:31, lwEndMfmaIndex:100  */
/*  numMfmaForLR:25, syncPlrMfmaIndex:102 , sync1LdsMfmaIndex:30 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(7)                               // wait for prior local read local write old=0, new=7 newLW=0 newLR=7 for iteration == 0
v_mfma_f32_16x16x16bf16_1k acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16bf16_1k acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
v_mfma_f32_16x16x16bf16_1k acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16bf16_1k acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuA_X2_I0+8:vgprValuA_X2_I0+8+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write
v_mfma_f32_16x16x16bf16_1k acc[16:19], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
v_mfma_f32_16x16x16bf16_1k acc[20:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
v_mfma_f32_16x16x16bf16_1k acc[24:27], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_X2_I0+12:vgprValuA_X2_I0+12+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
v_mfma_f32_16x16x16bf16_1k acc[28:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
v_mfma_f32_16x16x16bf16_1k acc[32:35], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB+0] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=2 iui=0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x16bf16_1k acc[36:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB+0] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=2 iui=0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16bf16_1k acc[40:43], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
v_mfma_f32_16x16x16bf16_1k acc[44:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16bf16_1k acc[48:51], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB+0] offset:576 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
v_mfma_f32_16x16x16bf16_1k acc[52:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB+0] offset:704 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=2 iui=0
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
v_mfma_f32_16x16x16bf16_1k acc[56:59], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB+0] offset:832 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=2 iui=0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
v_mfma_f32_16x16x16bf16_1k acc[60:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:16  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB+0] offset:960 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
v_mfma_f32_16x16x16bf16_1k acc[64:67], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
v_mfma_f32_16x16x16bf16_1k acc[68:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x16bf16_1k acc[72:75], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[76:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[80:83], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[84:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[88:91], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[92:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[92:95] // left value = acc[92+0:95+0]
/*  mfmaIndex:24  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[96:99], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[96:99] // left value = acc[96+0:99+0]
/*  mfmaIndex:25  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[100:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[100:103] // left value = acc[100+0:103+0]
/*  mfmaIndex:26  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[104:107], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[104:107] // left value = acc[104+0:107+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[108:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[108:111] // left value = acc[108+0:111+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[112:115], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[112:115] // left value = acc[112+0:115+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16bf16_1k acc[116:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[116:119] // left value = acc[116+0:119+0]
/*  mfmaIndex:30  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 2 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16bf16_1k acc[120:123], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+1], acc[120:123] // left value = acc[120+0:123+0]
/*  mfmaIndex:31  */
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
v_mfma_f32_16x16x16bf16_1k acc[124:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1], acc[124:127] // left value = acc[124+0:127+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */

/* iter 1 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:31, lwEndMfmaIndex:100  */
/*  numMfmaForLR:25, syncPlrMfmaIndex:102 , sync1LdsMfmaIndex:30 */
/*  mfmaIndex:32  */
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f32_16x16x16bf16_1k acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:33  */
v_mfma_f32_16x16x16bf16_1k acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:34  */
v_mfma_f32_16x16x16bf16_1k acc[8:11], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:35  */
v_mfma_f32_16x16x16bf16_1k acc[12:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:36  */
v_mfma_f32_16x16x16bf16_1k acc[16:19], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:37  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
v_mfma_f32_16x16x16bf16_1k acc[20:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:38  */
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f32_16x16x16bf16_1k acc[24:27], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:39  */
v_mfma_f32_16x16x16bf16_1k acc[28:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:40  */
v_mfma_f32_16x16x16bf16_1k acc[32:35], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:41  */
v_mfma_f32_16x16x16bf16_1k acc[36:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:42  */
v_mfma_f32_16x16x16bf16_1k acc[40:43], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:43  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
v_mfma_f32_16x16x16bf16_1k acc[44:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:44  */
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f32_16x16x16bf16_1k acc[48:51], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:45  */
v_mfma_f32_16x16x16bf16_1k acc[52:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:46  */
v_mfma_f32_16x16x16bf16_1k acc[56:59], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:47  */
v_mfma_f32_16x16x16bf16_1k acc[60:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:48  */
v_mfma_f32_16x16x16bf16_1k acc[64:67], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:49  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
v_mfma_f32_16x16x16bf16_1k acc[68:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:50  */
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f32_16x16x16bf16_1k acc[72:75], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:51  */
v_mfma_f32_16x16x16bf16_1k acc[76:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:52  */
v_mfma_f32_16x16x16bf16_1k acc[80:83], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x16bf16_1k acc[84:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:54  */
v_mfma_f32_16x16x16bf16_1k acc[88:91], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:55  */
v_mfma_f32_16x16x16bf16_1k acc[92:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[92:95] // left value = acc[92+0:95+0]
/*  mfmaIndex:56  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
v_mfma_f32_16x16x16bf16_1k acc[96:99], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[96:99] // left value = acc[96+0:99+0]
/*  mfmaIndex:57  */
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f32_16x16x16bf16_1k acc[100:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[100:103] // left value = acc[100+0:103+0]
/*  mfmaIndex:58  */
v_mfma_f32_16x16x16bf16_1k acc[104:107], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[104:107] // left value = acc[104+0:107+0]
/*  mfmaIndex:59  */
v_mfma_f32_16x16x16bf16_1k acc[108:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[108:111] // left value = acc[108+0:111+0]
/*  mfmaIndex:60  */
v_mfma_f32_16x16x16bf16_1k acc[112:115], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[112:115] // left value = acc[112+0:115+0]
/*  mfmaIndex:61  */
v_mfma_f32_16x16x16bf16_1k acc[116:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[116:119] // left value = acc[116+0:119+0]
/*  mfmaIndex:62  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4224 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4224
v_mfma_f32_16x16x16bf16_1k acc[120:123], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X0_I0+8+2+0:vgprValuA_X0_I0+8+2+0+1], acc[120:123] // left value = acc[120+0:123+0]
/*  mfmaIndex:63  */
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f32_16x16x16bf16_1k acc[124:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X0_I0+12+2+0:vgprValuA_X0_I0+12+2+0+1], acc[124:127] // left value = acc[124+0:127+0]

/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:31, lwEndMfmaIndex:100  */
/*  numMfmaForLR:25, syncPlrMfmaIndex:102 , sync1LdsMfmaIndex:30 */
/*  mfmaIndex:64  */
v_mfma_f32_16x16x16bf16_1k acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:65  */
v_mfma_f32_16x16x16bf16_1k acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:66  */
v_mfma_f32_16x16x16bf16_1k acc[8:11], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:67  */
v_mfma_f32_16x16x16bf16_1k acc[12:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:68  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:8448 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 8448
v_mfma_f32_16x16x16bf16_1k acc[16:19], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:69  */
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f32_16x16x16bf16_1k acc[20:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:70  */
v_mfma_f32_16x16x16bf16_1k acc[24:27], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:71  */
v_mfma_f32_16x16x16bf16_1k acc[28:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:72  */
v_mfma_f32_16x16x16bf16_1k acc[32:35], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:73  */
v_mfma_f32_16x16x16bf16_1k acc[36:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:74  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:12672 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 12672
v_mfma_f32_16x16x16bf16_1k acc[40:43], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:75  */
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f32_16x16x16bf16_1k acc[44:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:76  */
v_mfma_f32_16x16x16bf16_1k acc[48:51], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:77  */
v_mfma_f32_16x16x16bf16_1k acc[52:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:78  */
v_mfma_f32_16x16x16bf16_1k acc[56:59], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:79  */
v_mfma_f32_16x16x16bf16_1k acc[60:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:80  */
v_mfma_f32_16x16x16bf16_1k acc[64:67], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:81  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:16896 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 16896
v_mfma_f32_16x16x16bf16_1k acc[68:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:82  */
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
v_mfma_f32_16x16x16bf16_1k acc[72:75], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:83  */
v_mfma_f32_16x16x16bf16_1k acc[76:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:84  */
v_mfma_f32_16x16x16bf16_1k acc[80:83], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:85  */
v_mfma_f32_16x16x16bf16_1k acc[84:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:86  */
v_mfma_f32_16x16x16bf16_1k acc[88:91], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:87  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:21120 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 21120
v_mfma_f32_16x16x16bf16_1k acc[92:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[92:95] // left value = acc[92+0:95+0]
/*  mfmaIndex:88  */
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
v_mfma_f32_16x16x16bf16_1k acc[96:99], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[96:99] // left value = acc[96+0:99+0]
/*  mfmaIndex:89  */
v_mfma_f32_16x16x16bf16_1k acc[100:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[100:103] // left value = acc[100+0:103+0]
/*  mfmaIndex:90  */
v_mfma_f32_16x16x16bf16_1k acc[104:107], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[104:107] // left value = acc[104+0:107+0]
/*  mfmaIndex:91  */
v_mfma_f32_16x16x16bf16_1k acc[108:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[108:111] // left value = acc[108+0:111+0]
/*  mfmaIndex:92  */
v_mfma_f32_16x16x16bf16_1k acc[112:115], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[112:115] // left value = acc[112+0:115+0]
/*  mfmaIndex:93  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:25344 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 25344
v_mfma_f32_16x16x16bf16_1k acc[116:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[116:119] // left value = acc[116+0:119+0]
/*  mfmaIndex:94  */
buffer_load_dwordx4 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
v_mfma_f32_16x16x16bf16_1k acc[120:123], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+1], acc[120:123] // left value = acc[120+0:123+0]
/*  mfmaIndex:95  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16bf16_1k acc[124:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+12+0+0:vgprValuA_X2_I0+12+0+0+1], acc[124:127] // left value = acc[124+0:127+0]

/* iter 3 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:31, lwEndMfmaIndex:100  */
/*  numMfmaForLR:25, syncPlrMfmaIndex:102 , sync1LdsMfmaIndex:30 */
/*  mfmaIndex:96  */
v_mfma_f32_16x16x16bf16_1k acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:97  */
v_mfma_f32_16x16x16bf16_1k acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:98  */
v_mfma_f32_16x16x16bf16_1k acc[8:11], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:99  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:29568 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 29568
v_mfma_f32_16x16x16bf16_1k acc[12:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:100  */
buffer_load_dwordx4 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16bf16_1k acc[16:19], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:101  */
v_mfma_f32_16x16x16bf16_1k acc[20:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:102  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16bf16_1k acc[24:27], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:103  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[28:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:104  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[32:35], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:105  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[36:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:106  */
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[40:43], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:107  */
v_mfma_f32_16x16x16bf16_1k acc[44:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:108  */
v_mfma_f32_16x16x16bf16_1k acc[48:51], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:109  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[52:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:110  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[56:59], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:111  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[60:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[60:63] // left value = acc[60+0:63+0]
/*  mfmaIndex:112  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[64:67], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[64:67] // left value = acc[64+0:67+0]
/*  mfmaIndex:113  */
v_mfma_f32_16x16x16bf16_1k acc[68:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[68:71] // left value = acc[68+0:71+0]
/*  mfmaIndex:114  */
v_mfma_f32_16x16x16bf16_1k acc[72:75], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[72:75] // left value = acc[72+0:75+0]
/*  mfmaIndex:115  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[76:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[76:79] // left value = acc[76+0:79+0]
/*  mfmaIndex:116  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[80:83], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[80:83] // left value = acc[80+0:83+0]
/*  mfmaIndex:117  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[84:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[84:87] // left value = acc[84+0:87+0]
/*  mfmaIndex:118  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16bf16_1k acc[88:91], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[88:91] // left value = acc[88+0:91+0]
/*  mfmaIndex:119  */
v_mfma_f32_16x16x16bf16_1k acc[92:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[92:95] // left value = acc[92+0:95+0]
/*  mfmaIndex:120  */
v_mfma_f32_16x16x16bf16_1k acc[96:99], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[96:99] // left value = acc[96+0:99+0]
/*  mfmaIndex:121  */
v_mfma_f32_16x16x16bf16_1k acc[100:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[100:103] // left value = acc[100+0:103+0]
/*  mfmaIndex:122  */
v_mfma_f32_16x16x16bf16_1k acc[104:107], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[104:107] // left value = acc[104+0:107+0]
/*  mfmaIndex:123  */
v_mfma_f32_16x16x16bf16_1k acc[108:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[108:111] // left value = acc[108+0:111+0]
/*  mfmaIndex:124  */
v_mfma_f32_16x16x16bf16_1k acc[112:115], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[112:115] // left value = acc[112+0:115+0]
/*  mfmaIndex:125  */
v_mfma_f32_16x16x16bf16_1k acc[116:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[116:119] // left value = acc[116+0:119+0]
/*  mfmaIndex:126  */
v_mfma_f32_16x16x16bf16_1k acc[120:123], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X2_I0+8+2+0:vgprValuA_X2_I0+8+2+0+1], acc[120:123] // left value = acc[120+0:123+0]
/*  mfmaIndex:127  */
v_mfma_f32_16x16x16bf16_1k acc[124:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X2_I0+12+2+0:vgprValuA_X2_I0+12+2+0+1], acc[124:127] // left value = acc[124+0:127+0]

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
