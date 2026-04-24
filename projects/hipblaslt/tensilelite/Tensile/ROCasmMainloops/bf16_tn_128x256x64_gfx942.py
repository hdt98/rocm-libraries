from Tensile.Components.ROCasmRegistry import RegisterROCasmMainloop
from rocasm.block import Block
from rocasm.regs import VgprArray, AccArray, SgprArray
from rocasm.instructions import buffer_load_dwordx4, ds_read_b128, ds_write_b128, vmfma_f32_16x16x16bf16_1k
from rocisa.container import DSModifiers, MUBUFModifiers, sgpr


@RegisterROCasmMainloop(
    macro_tile_0=128, macro_tile_1=256, depth_u=64,
    matrix_inst=[16, 16, 16, 1],
    transpose_a=True, transpose_b=False,
)
def rocasm_main_loop():
    """ROCasm translation of the main loop.

    Auto-generated from the assembly main loop by asm_to_rocasm.
    Edit this function to modify the main loop programmatically.
    """
    block = Block(
        A0=VgprArray(base=16, count=16),
        A2=VgprArray(base=32, count=16),
        B0=VgprArray(base=48, count=32),
        B2=VgprArray(base=80, count=32),
        G2LA=VgprArray(base=112, count=16),
        G2LB=VgprArray(base=128, count=32),
        GlobalReadOffsetA=VgprArray(base=0, count=4),
        GlobalReadOffsetB=VgprArray(base=4, count=8),
        LocalReadAddrA=VgprArray(base=14, count=1),
        LocalReadAddrB=VgprArray(base=15, count=1),
        LocalWriteAddrA=VgprArray(base=12, count=1),
        LocalWriteAddrB=VgprArray(base=13, count=1),
        SrdA=SgprArray(base=48, count=4),
        SrdB=SgprArray(base=52, count=4),
        Acc=AccArray(base=0, count=128),
    )

    A0 = block.A0
    A2 = block.A2
    B0 = block.B0
    B2 = block.B2
    G2LA = block.G2LA
    G2LB = block.G2LB
    GlobalReadOffsetA = block.GlobalReadOffsetA
    GlobalReadOffsetB = block.GlobalReadOffsetB
    LocalReadAddrA = block.LocalReadAddrA
    LocalReadAddrB = block.LocalReadAddrB
    LocalWriteAddrA = block.LocalWriteAddrA
    LocalWriteAddrB = block.LocalWriteAddrB
    SrdA = block.SrdA
    SrdB = block.SrdB
    Acc = block.Acc
    label = block.label
    s_add_u32 = block.s_add_u32
    s_addc_u32 = block.s_addc_u32
    s_barrier = block.s_barrier
    s_cbranch_scc0 = block.s_cbranch_scc0
    s_cmp_eq_i32 = block.s_cmp_eq_i32
    s_cmp_eq_u32 = block.s_cmp_eq_u32
    s_cselect_b32 = block.s_cselect_b32
    s_sub_u32 = block.s_sub_u32
    s_subb_u32 = block.s_subb_u32
    s_waitcnt = block.s_waitcnt
    
    label("label_LoopBeginL")
    s_waitcnt(dscnt=7)
    Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[0:2], Acc[0:4])
    A2[0:4] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=64))
    s_cmp_eq_u32(src0=sgpr(12), src1=sgpr(47))
    Acc[4:8] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[4:6], Acc[4:8])
    B2[0:4] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=64))
    s_cselect_b32(dst=sgpr(66), src0=sgpr(60), src1=sgpr(64))
    Acc[8:12] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[8:10], Acc[8:12])
    A2[4:8] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=192))
    s_cselect_b32(dst=sgpr(67), src0=sgpr(61), src1=0)
    Acc[12:16] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[12:14], Acc[12:16])
    A2[8:12] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=320))
    s_add_u32(dst=sgpr(48), src0=sgpr(48), src1=sgpr(66))
    s_waitcnt(dscnt=4)
    Acc[16:20] = vmfma_f32_16x16x16bf16_1k(B0[4:6], A0[0:2], Acc[16:20])
    s_addc_u32(dst=sgpr(49), src0=sgpr(49), src1=sgpr(67))
    Acc[20:24] = vmfma_f32_16x16x16bf16_1k(B0[4:6], A0[4:6], Acc[20:24])
    s_sub_u32(dst=sgpr(56), src0=sgpr(56), src1=sgpr(66))
    Acc[24:28] = vmfma_f32_16x16x16bf16_1k(B0[4:6], A0[8:10], Acc[24:28])
    A2[12:16] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=448))
    s_subb_u32(dst=sgpr(57), src0=sgpr(57), src1=sgpr(67))
    Acc[28:32] = vmfma_f32_16x16x16bf16_1k(B0[4:6], A0[12:14], Acc[28:32])
    B2[4:8] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=192))
    s_cmp_eq_u32(src0=sgpr(57), src1=0)
    Acc[32:36] = vmfma_f32_16x16x16bf16_1k(B0[8:10], A0[0:2], Acc[32:36])
    B2[8:12] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=320))
    s_cselect_b32(dst=sgpr(50), src0=sgpr(56), src1=4294967295)
    Acc[36:40] = vmfma_f32_16x16x16bf16_1k(B0[8:10], A0[4:6], Acc[36:40])
    B2[12:16] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=448))
    s_cmp_eq_u32(src0=sgpr(12), src1=sgpr(47))
    Acc[40:44] = vmfma_f32_16x16x16bf16_1k(B0[8:10], A0[8:10], Acc[40:44])
    s_cselect_b32(dst=sgpr(66), src0=sgpr(62), src1=sgpr(65))
    Acc[44:48] = vmfma_f32_16x16x16bf16_1k(B0[8:10], A0[12:14], Acc[44:48])
    s_cselect_b32(dst=sgpr(67), src0=sgpr(63), src1=0)
    Acc[48:52] = vmfma_f32_16x16x16bf16_1k(B0[12:14], A0[0:2], Acc[48:52])
    B2[16:20] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=576))
    s_add_u32(dst=sgpr(52), src0=sgpr(52), src1=sgpr(66))
    Acc[52:56] = vmfma_f32_16x16x16bf16_1k(B0[12:14], A0[4:6], Acc[52:56])
    B2[20:24] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=704))
    s_addc_u32(dst=sgpr(53), src0=sgpr(53), src1=sgpr(67))
    Acc[56:60] = vmfma_f32_16x16x16bf16_1k(B0[12:14], A0[8:10], Acc[56:60])
    B2[24:28] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=832))
    s_sub_u32(dst=sgpr(58), src0=sgpr(58), src1=sgpr(66))
    Acc[60:64] = vmfma_f32_16x16x16bf16_1k(B0[12:14], A0[12:14], Acc[60:64])
    B2[28:32] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=960))
    s_subb_u32(dst=sgpr(59), src0=sgpr(59), src1=sgpr(67))
    Acc[64:68] = vmfma_f32_16x16x16bf16_1k(B0[16:18], A0[0:2], Acc[64:68])
    s_cmp_eq_u32(src0=sgpr(59), src1=0)
    Acc[68:72] = vmfma_f32_16x16x16bf16_1k(B0[16:18], A0[4:6], Acc[68:72])
    s_cselect_b32(dst=sgpr(54), src0=sgpr(58), src1=4294967295)
    Acc[72:76] = vmfma_f32_16x16x16bf16_1k(B0[16:18], A0[8:10], Acc[72:76])
    Acc[76:80] = vmfma_f32_16x16x16bf16_1k(B0[16:18], A0[12:14], Acc[76:80])
    Acc[80:84] = vmfma_f32_16x16x16bf16_1k(B0[20:22], A0[0:2], Acc[80:84])
    Acc[84:88] = vmfma_f32_16x16x16bf16_1k(B0[20:22], A0[4:6], Acc[84:88])
    Acc[88:92] = vmfma_f32_16x16x16bf16_1k(B0[20:22], A0[8:10], Acc[88:92])
    Acc[92:96] = vmfma_f32_16x16x16bf16_1k(B0[20:22], A0[12:14], Acc[92:96])
    Acc[96:100] = vmfma_f32_16x16x16bf16_1k(B0[24:26], A0[0:2], Acc[96:100])
    Acc[100:104] = vmfma_f32_16x16x16bf16_1k(B0[24:26], A0[4:6], Acc[100:104])
    Acc[104:108] = vmfma_f32_16x16x16bf16_1k(B0[24:26], A0[8:10], Acc[104:108])
    Acc[108:112] = vmfma_f32_16x16x16bf16_1k(B0[24:26], A0[12:14], Acc[108:112])
    Acc[112:116] = vmfma_f32_16x16x16bf16_1k(B0[28:30], A0[0:2], Acc[112:116])
    Acc[116:120] = vmfma_f32_16x16x16bf16_1k(B0[28:30], A0[4:6], Acc[116:120])
    s_waitcnt(dscnt=0)
    s_barrier()
    Acc[120:124] = vmfma_f32_16x16x16bf16_1k(B0[28:30], A0[8:10], Acc[120:124])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrA[0:1], G2LA[0:4])
    Acc[124:128] = vmfma_f32_16x16x16bf16_1k(B0[28:30], A0[12:14], Acc[124:128])
    G2LA[0:4] = buffer_load_dwordx4(GlobalReadOffsetA[0:1], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B0[2:4], A0[2:4], Acc[0:4])
    Acc[4:8] = vmfma_f32_16x16x16bf16_1k(B0[2:4], A0[6:8], Acc[4:8])
    Acc[8:12] = vmfma_f32_16x16x16bf16_1k(B0[2:4], A0[10:12], Acc[8:12])
    Acc[12:16] = vmfma_f32_16x16x16bf16_1k(B0[2:4], A0[14:16], Acc[12:16])
    Acc[16:20] = vmfma_f32_16x16x16bf16_1k(B0[6:8], A0[2:4], Acc[16:20])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrA[0:1], G2LA[4:8], ds=DSModifiers(offset=4352))
    Acc[20:24] = vmfma_f32_16x16x16bf16_1k(B0[6:8], A0[6:8], Acc[20:24])
    G2LA[4:8] = buffer_load_dwordx4(GlobalReadOffsetA[1:2], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[24:28] = vmfma_f32_16x16x16bf16_1k(B0[6:8], A0[10:12], Acc[24:28])
    Acc[28:32] = vmfma_f32_16x16x16bf16_1k(B0[6:8], A0[14:16], Acc[28:32])
    Acc[32:36] = vmfma_f32_16x16x16bf16_1k(B0[10:12], A0[2:4], Acc[32:36])
    Acc[36:40] = vmfma_f32_16x16x16bf16_1k(B0[10:12], A0[6:8], Acc[36:40])
    Acc[40:44] = vmfma_f32_16x16x16bf16_1k(B0[10:12], A0[10:12], Acc[40:44])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrA[0:1], G2LA[8:12], ds=DSModifiers(offset=8704))
    Acc[44:48] = vmfma_f32_16x16x16bf16_1k(B0[10:12], A0[14:16], Acc[44:48])
    G2LA[8:12] = buffer_load_dwordx4(GlobalReadOffsetA[2:3], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[48:52] = vmfma_f32_16x16x16bf16_1k(B0[14:16], A0[2:4], Acc[48:52])
    Acc[52:56] = vmfma_f32_16x16x16bf16_1k(B0[14:16], A0[6:8], Acc[52:56])
    Acc[56:60] = vmfma_f32_16x16x16bf16_1k(B0[14:16], A0[10:12], Acc[56:60])
    Acc[60:64] = vmfma_f32_16x16x16bf16_1k(B0[14:16], A0[14:16], Acc[60:64])
    Acc[64:68] = vmfma_f32_16x16x16bf16_1k(B0[18:20], A0[2:4], Acc[64:68])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrA[0:1], G2LA[12:16], ds=DSModifiers(offset=13056))
    Acc[68:72] = vmfma_f32_16x16x16bf16_1k(B0[18:20], A0[6:8], Acc[68:72])
    G2LA[12:16] = buffer_load_dwordx4(GlobalReadOffsetA[3:4], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[72:76] = vmfma_f32_16x16x16bf16_1k(B0[18:20], A0[10:12], Acc[72:76])
    Acc[76:80] = vmfma_f32_16x16x16bf16_1k(B0[18:20], A0[14:16], Acc[76:80])
    Acc[80:84] = vmfma_f32_16x16x16bf16_1k(B0[22:24], A0[2:4], Acc[80:84])
    Acc[84:88] = vmfma_f32_16x16x16bf16_1k(B0[22:24], A0[6:8], Acc[84:88])
    Acc[88:92] = vmfma_f32_16x16x16bf16_1k(B0[22:24], A0[10:12], Acc[88:92])
    Acc[92:96] = vmfma_f32_16x16x16bf16_1k(B0[22:24], A0[14:16], Acc[92:96])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[0:4])
    Acc[96:100] = vmfma_f32_16x16x16bf16_1k(B0[26:28], A0[2:4], Acc[96:100])
    G2LB[0:4] = buffer_load_dwordx4(GlobalReadOffsetB[0:1], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[100:104] = vmfma_f32_16x16x16bf16_1k(B0[26:28], A0[6:8], Acc[100:104])
    Acc[104:108] = vmfma_f32_16x16x16bf16_1k(B0[26:28], A0[10:12], Acc[104:108])
    Acc[108:112] = vmfma_f32_16x16x16bf16_1k(B0[26:28], A0[14:16], Acc[108:112])
    Acc[112:116] = vmfma_f32_16x16x16bf16_1k(B0[30:32], A0[2:4], Acc[112:116])
    Acc[116:120] = vmfma_f32_16x16x16bf16_1k(B0[30:32], A0[6:8], Acc[116:120])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[4:8], ds=DSModifiers(offset=4224))
    Acc[120:124] = vmfma_f32_16x16x16bf16_1k(B0[30:32], A0[10:12], Acc[120:124])
    G2LB[4:8] = buffer_load_dwordx4(GlobalReadOffsetB[1:2], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[124:128] = vmfma_f32_16x16x16bf16_1k(B0[30:32], A0[14:16], Acc[124:128])
    Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B2[0:2], A2[0:2], Acc[0:4])
    Acc[4:8] = vmfma_f32_16x16x16bf16_1k(B2[0:2], A2[4:6], Acc[4:8])
    Acc[8:12] = vmfma_f32_16x16x16bf16_1k(B2[0:2], A2[8:10], Acc[8:12])
    Acc[12:16] = vmfma_f32_16x16x16bf16_1k(B2[0:2], A2[12:14], Acc[12:16])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[8:12], ds=DSModifiers(offset=8448))
    Acc[16:20] = vmfma_f32_16x16x16bf16_1k(B2[4:6], A2[0:2], Acc[16:20])
    G2LB[8:12] = buffer_load_dwordx4(GlobalReadOffsetB[2:3], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[20:24] = vmfma_f32_16x16x16bf16_1k(B2[4:6], A2[4:6], Acc[20:24])
    Acc[24:28] = vmfma_f32_16x16x16bf16_1k(B2[4:6], A2[8:10], Acc[24:28])
    Acc[28:32] = vmfma_f32_16x16x16bf16_1k(B2[4:6], A2[12:14], Acc[28:32])
    Acc[32:36] = vmfma_f32_16x16x16bf16_1k(B2[8:10], A2[0:2], Acc[32:36])
    Acc[36:40] = vmfma_f32_16x16x16bf16_1k(B2[8:10], A2[4:6], Acc[36:40])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[12:16], ds=DSModifiers(offset=12672))
    Acc[40:44] = vmfma_f32_16x16x16bf16_1k(B2[8:10], A2[8:10], Acc[40:44])
    G2LB[12:16] = buffer_load_dwordx4(GlobalReadOffsetB[3:4], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[44:48] = vmfma_f32_16x16x16bf16_1k(B2[8:10], A2[12:14], Acc[44:48])
    Acc[48:52] = vmfma_f32_16x16x16bf16_1k(B2[12:14], A2[0:2], Acc[48:52])
    Acc[52:56] = vmfma_f32_16x16x16bf16_1k(B2[12:14], A2[4:6], Acc[52:56])
    Acc[56:60] = vmfma_f32_16x16x16bf16_1k(B2[12:14], A2[8:10], Acc[56:60])
    Acc[60:64] = vmfma_f32_16x16x16bf16_1k(B2[12:14], A2[12:14], Acc[60:64])
    Acc[64:68] = vmfma_f32_16x16x16bf16_1k(B2[16:18], A2[0:2], Acc[64:68])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[16:20], ds=DSModifiers(offset=16896))
    Acc[68:72] = vmfma_f32_16x16x16bf16_1k(B2[16:18], A2[4:6], Acc[68:72])
    G2LB[16:20] = buffer_load_dwordx4(GlobalReadOffsetB[4:5], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[72:76] = vmfma_f32_16x16x16bf16_1k(B2[16:18], A2[8:10], Acc[72:76])
    Acc[76:80] = vmfma_f32_16x16x16bf16_1k(B2[16:18], A2[12:14], Acc[76:80])
    Acc[80:84] = vmfma_f32_16x16x16bf16_1k(B2[20:22], A2[0:2], Acc[80:84])
    Acc[84:88] = vmfma_f32_16x16x16bf16_1k(B2[20:22], A2[4:6], Acc[84:88])
    Acc[88:92] = vmfma_f32_16x16x16bf16_1k(B2[20:22], A2[8:10], Acc[88:92])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[20:24], ds=DSModifiers(offset=21120))
    Acc[92:96] = vmfma_f32_16x16x16bf16_1k(B2[20:22], A2[12:14], Acc[92:96])
    G2LB[20:24] = buffer_load_dwordx4(GlobalReadOffsetB[5:6], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[96:100] = vmfma_f32_16x16x16bf16_1k(B2[24:26], A2[0:2], Acc[96:100])
    Acc[100:104] = vmfma_f32_16x16x16bf16_1k(B2[24:26], A2[4:6], Acc[100:104])
    Acc[104:108] = vmfma_f32_16x16x16bf16_1k(B2[24:26], A2[8:10], Acc[104:108])
    Acc[108:112] = vmfma_f32_16x16x16bf16_1k(B2[24:26], A2[12:14], Acc[108:112])
    Acc[112:116] = vmfma_f32_16x16x16bf16_1k(B2[28:30], A2[0:2], Acc[112:116])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[24:28], ds=DSModifiers(offset=25344))
    Acc[116:120] = vmfma_f32_16x16x16bf16_1k(B2[28:30], A2[4:6], Acc[116:120])
    G2LB[24:28] = buffer_load_dwordx4(GlobalReadOffsetB[6:7], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[120:124] = vmfma_f32_16x16x16bf16_1k(B2[28:30], A2[8:10], Acc[120:124])
    Acc[124:128] = vmfma_f32_16x16x16bf16_1k(B2[28:30], A2[12:14], Acc[124:128])
    Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B2[2:4], A2[2:4], Acc[0:4])
    Acc[4:8] = vmfma_f32_16x16x16bf16_1k(B2[2:4], A2[6:8], Acc[4:8])
    Acc[8:12] = vmfma_f32_16x16x16bf16_1k(B2[2:4], A2[10:12], Acc[8:12])
    s_waitcnt(vlcnt=11)
    ds_write_b128(LocalWriteAddrB[0:1], G2LB[28:32], ds=DSModifiers(offset=29568))
    Acc[12:16] = vmfma_f32_16x16x16bf16_1k(B2[2:4], A2[14:16], Acc[12:16])
    G2LB[28:32] = buffer_load_dwordx4(GlobalReadOffsetB[7:8], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True))
    Acc[16:20] = vmfma_f32_16x16x16bf16_1k(B2[6:8], A2[2:4], Acc[16:20])
    Acc[20:24] = vmfma_f32_16x16x16bf16_1k(B2[6:8], A2[6:8], Acc[20:24])
    s_waitcnt(dscnt=0)
    s_barrier()
    Acc[24:28] = vmfma_f32_16x16x16bf16_1k(B2[6:8], A2[10:12], Acc[24:28])
    A0[0:4] = ds_read_b128(LocalReadAddrA[0:1])
    Acc[28:32] = vmfma_f32_16x16x16bf16_1k(B2[6:8], A2[14:16], Acc[28:32])
    B0[0:4] = ds_read_b128(LocalReadAddrB[0:1])
    Acc[32:36] = vmfma_f32_16x16x16bf16_1k(B2[10:12], A2[2:4], Acc[32:36])
    A0[4:8] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=128))
    Acc[36:40] = vmfma_f32_16x16x16bf16_1k(B2[10:12], A2[6:8], Acc[36:40])
    A0[8:12] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=256))
    Acc[40:44] = vmfma_f32_16x16x16bf16_1k(B2[10:12], A2[10:12], Acc[40:44])
    Acc[44:48] = vmfma_f32_16x16x16bf16_1k(B2[10:12], A2[14:16], Acc[44:48])
    Acc[48:52] = vmfma_f32_16x16x16bf16_1k(B2[14:16], A2[2:4], Acc[48:52])
    A0[12:16] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=384))
    Acc[52:56] = vmfma_f32_16x16x16bf16_1k(B2[14:16], A2[6:8], Acc[52:56])
    B0[4:8] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=128))
    Acc[56:60] = vmfma_f32_16x16x16bf16_1k(B2[14:16], A2[10:12], Acc[56:60])
    B0[8:12] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=256))
    Acc[60:64] = vmfma_f32_16x16x16bf16_1k(B2[14:16], A2[14:16], Acc[60:64])
    B0[12:16] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=384))
    Acc[64:68] = vmfma_f32_16x16x16bf16_1k(B2[18:20], A2[2:4], Acc[64:68])
    Acc[68:72] = vmfma_f32_16x16x16bf16_1k(B2[18:20], A2[6:8], Acc[68:72])
    Acc[72:76] = vmfma_f32_16x16x16bf16_1k(B2[18:20], A2[10:12], Acc[72:76])
    B0[16:20] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=512))
    Acc[76:80] = vmfma_f32_16x16x16bf16_1k(B2[18:20], A2[14:16], Acc[76:80])
    B0[20:24] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=640))
    Acc[80:84] = vmfma_f32_16x16x16bf16_1k(B2[22:24], A2[2:4], Acc[80:84])
    B0[24:28] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=768))
    Acc[84:88] = vmfma_f32_16x16x16bf16_1k(B2[22:24], A2[6:8], Acc[84:88])
    B0[28:32] = ds_read_b128(LocalReadAddrB[0:1], ds=DSModifiers(offset=896))
    Acc[88:92] = vmfma_f32_16x16x16bf16_1k(B2[22:24], A2[10:12], Acc[88:92])
    Acc[92:96] = vmfma_f32_16x16x16bf16_1k(B2[22:24], A2[14:16], Acc[92:96])
    Acc[96:100] = vmfma_f32_16x16x16bf16_1k(B2[26:28], A2[2:4], Acc[96:100])
    Acc[100:104] = vmfma_f32_16x16x16bf16_1k(B2[26:28], A2[6:8], Acc[100:104])
    Acc[104:108] = vmfma_f32_16x16x16bf16_1k(B2[26:28], A2[10:12], Acc[104:108])
    Acc[108:112] = vmfma_f32_16x16x16bf16_1k(B2[26:28], A2[14:16], Acc[108:112])
    Acc[112:116] = vmfma_f32_16x16x16bf16_1k(B2[30:32], A2[2:4], Acc[112:116])
    Acc[116:120] = vmfma_f32_16x16x16bf16_1k(B2[30:32], A2[6:8], Acc[116:120])
    Acc[120:124] = vmfma_f32_16x16x16bf16_1k(B2[30:32], A2[10:12], Acc[120:124])
    Acc[124:128] = vmfma_f32_16x16x16bf16_1k(B2[30:32], A2[14:16], Acc[124:128])
    s_sub_u32(dst=sgpr(12), src0=sgpr(12), src1=1)
    s_cmp_eq_i32(src0=sgpr(12), src1=0x2)
    s_cbranch_scc0(labelName="label_LoopBeginL")

    return block
