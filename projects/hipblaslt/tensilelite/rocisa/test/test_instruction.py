################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import rocisa
from rocisa.container import sgpr, vgpr
from copy import deepcopy
import math

def test_instruction_common():
    from rocisa.instruction import SMovB32

    inst = SMovB32(dst=sgpr(1), src=1.6)
    assert str(inst) == "s_mov_b32 s1, 1.6000000000000001\n"
    assert str(inst.dst) == "s1"
    # Due to conversion between C++ and Python
    assert math.isclose(inst.srcs[0], 1.6, abs_tol=0.000001)
    # You cannot use srcs[0] = 1.3 because it may break the C++ memory layout
    # so nanobind disabled it
    inst.srcs = [1.3]
    assert math.isclose(inst.srcs[0], 1.3, abs_tol=0.000001)
    inst.setSrc(0, 1.4)
    assert math.isclose(inst.srcs[0], 1.4, abs_tol=0.000001)

    inst2 = deepcopy(inst)
    inst.setSrc(0, 2.0)
    assert math.isclose(inst2.srcs[0], 1.4, abs_tol=0.000001)

    from rocisa.instruction import InstructionInputVector
    iiv = InstructionInputVector()
    iiv.append(1.0)
    iiv.append("hello")
    iiv.append(sgpr(1))
    assert str(iiv[2]) == "s1"

    from rocisa.code import Module
    from rocisa.container import vgpr
    from rocisa.instruction import BufferLoadB64
    module = Module("Test")
    module.add(BufferLoadB64(vgpr(1), vgpr(2), vgpr(3), 3))
    assert rocisa.countGlobalRead(module) == 1

def test_instruction_cvt():
    from rocisa.instruction import VCvtF16toF32, VCvtF32toF16, VCvtF32toU32, VCvtU32toF32, \
        VCvtI32toF32, VCvtF32toI32, VCvtFP8toF32, VCvtBF8toF32, VCvtPkFP8toF32, VCvtPkBF8toF32, \
            VCvtPkF32toFP8, VCvtPkF32toBF8, VCvtSRF32toFP8, VCvtSRF32toBF8

    # Test VCvtF16toF32
    inst = VCvtF16toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_f16 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toF16
    inst = VCvtF32toF16(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f16_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toU32
    inst = VCvtF32toU32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_u32_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtU32toF32
    inst = VCvtU32toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_u32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtI32toF32
    inst = VCvtI32toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_i32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtF32toI32
    inst = VCvtF32toI32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_i32_f32 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtFP8toF32
    inst = VCvtFP8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_fp8 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtBF8toF32
    inst = VCvtBF8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_f32_bf8 v1, v2                               // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkFP8toF32
    inst = VCvtPkFP8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_pk_f32_fp8 v1, v2                            // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkBF8toF32
    inst = VCvtPkBF8toF32(dst=vgpr(1), src=vgpr(2), comment="test comment")
    assert str(inst) == "v_cvt_pk_f32_bf8 v1, v2                            // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert inst.comment == "test comment"

    # Test VCvtPkF32toFP8
    inst = VCvtPkF32toFP8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_pk_fp8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtPkF32toBF8
    inst = VCvtPkF32toBF8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_pk_bf8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtSRF32toFP8
    inst = VCvtSRF32toFP8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_sr_fp8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

    # Test VCvtSRF32toBF8
    inst = VCvtSRF32toBF8(dst=vgpr(1), src0=vgpr(2), src1=vgpr(3), comment="test comment")
    assert str(inst) == "v_cvt_sr_bf8_f32 v1, v2, v3                        // test comment\n"
    assert str(inst.dst) == "v1"
    assert str(inst.srcs[0]) == "v2"
    assert str(inst.srcs[1]) == "v3"
    assert inst.comment == "test comment"

def test_instruction_tdm():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    inst = TensorLoadToLds(
        group0=sgpr(0, 4),
        group1=sgpr(0, 8),
        group2=sgpr(0, 4),
        group3=sgpr(0, 4),
        comment=""
    )
    assert str(inst) == "tensor_load_to_lds s[0:3], s[0:7], s[0:3], s[0:3]\n"

    try:
        TensorLoadToLds(
            group0=vgpr(0, 4),
            group1=sgpr(0, 8),
            group2=sgpr(0, 4),
            group3=sgpr(0, 4),
            comment=""
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        pass

def test_instruction_tdm_2_sgprs():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    inst = TensorLoadToLds(
        group0=sgpr(0, 4),
        group1=sgpr(0, 8),
        group2=None,
        group3=None,
        comment=""
    )
    assert str(inst) == "tensor_load_to_lds s[0:3], s[0:7]\n"

    try:
        TensorLoadToLds(
            group0=vgpr(0, 4),
            group1=sgpr(0, 8),
            group2=None,
            group3=None,
            comment=""
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        pass

def test_instruction_ds_store_b192():
    """Tests for the DSStoreB192 compound instruction.
    DSStoreB192 is a 192-bit (6 dword) DS store that decomposes into two
    sub-instructions at assembly time:
      1. ds_write_b128 (lower 4 dwords)  [ds_store_b128 for ISA >= gfx11]
      2. ds_write_b64  (upper 2 dwords)  [ds_store_b64  for ISA >= gfx11]
    The second sub-instruction has its DS offset incremented by 16 bytes.
    Named registers (vgpr("name", n)) must be used for the source operand
    because getArgStr2() manipulates regName->offsets to split the register
    range. Flat-indexed registers (vgpr(int, n)) would crash because their
    regName is nullopt.
    """
    from rocisa.instruction import DSStoreB192
    from rocisa.container import DSModifiers
    from rocisa.code import Module
    from copy import deepcopy
    # =========================================================================
    # Test 1: Basic construction with named registers, no modifiers, no comment
    # =========================================================================
    # dstAddr: flat-indexed single VGPR for the LDS address
    # src: named register with 6 dwords (192 bits)
    inst = DSStoreB192(
        dstAddr=vgpr(0),           # v0 -- LDS address register
        src=vgpr("SrcData+0", 6), # v[vgprSrcData+0:vgprSrcData+0+5] -- 6 dwords of data
    )
    output = str(inst)
    # The output should be two lines (two sub-instructions), each ending with \n
    lines = output.strip().split('\n')
    assert len(lines) == 2, \
        f"DSStoreB192 should emit exactly 2 sub-instructions, got {len(lines)}"
    # Line 1: ds_write_b128 with the lower 4 dwords of src
    # ISA (9,0,10) < gfx11, so legacy mnemonic "ds_write_b128" is used
    assert "ds_write_b128" in lines[0], \
        f"First sub-inst should be ds_write_b128, got: {lines[0]}"
    # dstAddr (v0) should appear in the arguments
    assert "v0" in lines[0], \
        f"First sub-inst should reference dstAddr v0, got: {lines[0]}"
    # Lower 4 dwords: vgprSrcData+0 with regNum=4 -> v[vgprSrcData+0:vgprSrcData+0+3]
    assert "vgprSrcData+0:vgprSrcData+0+3" in lines[0], \
        f"First sub-inst should use lower 4 dwords, got: {lines[0]}"
    # Line 2: ds_write_b64 with the upper 2 dwords of src, offset +16
    assert "ds_write_b64" in lines[1], \
        f"Second sub-inst should be ds_write_b64, got: {lines[1]}"
    assert "v0" in lines[1], \
        f"Second sub-inst should reference dstAddr v0, got: {lines[1]}"
    # Upper 2 dwords: offset shifted by 4 -> vgprSrcData+4, regNum=2
    assert "vgprSrcData+4:vgprSrcData+4+1" in lines[1], \
        f"Second sub-inst should use upper 2 dwords, got: {lines[1]}"
    # Offset should be 16 (4 dwords * 4 bytes = 16 bytes from base)
    assert "offset:16" in lines[1], \
        f"Second sub-inst should have offset:16, got: {lines[1]}"
    # =========================================================================
    # Test 2: Construction with DS modifiers (explicit offset)
    # =========================================================================
    # When a base offset is provided, the first sub-instruction uses it directly,
    # and the second sub-instruction adds 16 to it.
    inst_with_offset = DSStoreB192(
        dstAddr=vgpr(2),
        src=vgpr("StoreVal+0", 6),
        ds=DSModifiers(offset=32),  # base offset = 32 bytes
    )
    output2 = str(inst_with_offset)
    lines2 = output2.strip().split('\n')
    assert len(lines2) == 2
    # Line 1: should carry the original offset 32
    assert "offset:32" in lines2[0], \
        f"First sub-inst should have offset:32, got: {lines2[0]}"
    # Line 2: should carry offset 32 + 16 = 48
    assert "offset:48" in lines2[1], \
        f"Second sub-inst should have offset:48, got: {lines2[1]}"
    # =========================================================================
    # Test 3: Construction with comment
    # =========================================================================
    # Comments should appear on both sub-instruction lines
    inst_commented = DSStoreB192(
        dstAddr=vgpr(0),
        src=vgpr("D+0", 6),
        comment="store 192b"
    )
    output3 = str(inst_commented)
    lines3 = output3.strip().split('\n')
    assert len(lines3) == 2
    # Both lines should contain the comment
    assert "store 192b" in lines3[0], \
        f"Comment should appear in first line, got: {lines3[0]}"
    assert "store 192b" in lines3[1], \
        f"Comment should appear in second line, got: {lines3[1]}"
    # Verify comment field is accessible and correct
    assert inst_commented.comment == "store 192b"
    # =========================================================================
    # Test 4: Static issueLatency()
    # =========================================================================
    assert DSStoreB192.issueLatency() == 8, \
        f"issueLatency should be 8, got {DSStoreB192.issueLatency()}"
    # =========================================================================
    # Test 5: deepcopy independence
    # =========================================================================
    inst_orig = DSStoreB192(
        dstAddr=vgpr(0),
        src=vgpr("CopyTest+0", 6),
        comment="original"
    )
    inst_copy = deepcopy(inst_orig)
    # The copy should produce identical assembly output
    assert str(inst_orig) == str(inst_copy), \
        "deepcopy should produce identical assembly output"
    # But they should be distinct objects
    assert inst_orig is not inst_copy, \
        "deepcopy should produce a new object"
    # Mutating the original's comment should not affect the copy
    inst_orig.comment = "modified"
    assert inst_copy.comment == "original", \
        "deepcopy should be independent: modifying original must not affect copy"
    # Verify the assembly output diverges after mutation
    assert str(inst_orig) != str(inst_copy), \
        "After comment mutation, assembly outputs should differ"
    # =========================================================================
    # Test 6: Module integration and counting functions
    # =========================================================================
    module = Module("DSStoreB192Test")
    inst_for_module = DSStoreB192(
        dstAddr=vgpr(0),
        src=vgpr("ModData+0", 6),
    )
    module.add(inst_for_module)
    # countLocalWrite counts any LocalWriteInstruction-derived instruction
    assert rocisa.countLocalWrite(module) == 1, \
        f"countLocalWrite should be 1, got {rocisa.countLocalWrite(module)}"
    # countDSStoreB192 counts only DSStoreB192 instructions specifically
    assert rocisa.countDSStoreB192(module) == 1, \
        f"countDSStoreB192 should be 1, got {rocisa.countDSStoreB192(module)}"
    # countWeightedLocalWrite gives weight 2 for DSStoreB192 because it
    # decomposes into 2 sub-instructions at assembly time
    assert rocisa.countWeightedLocalWrite(module) == 2, \
        f"countWeightedLocalWrite should be 2, got {rocisa.countWeightedLocalWrite(module)}"
    # =========================================================================
    # Test 7: getParams() verification
    # =========================================================================
    # DSStoreInstruction::getParams() returns {dstAddr, src0, src1}.
    # For DSStoreB192, src1 is nullptr (only dstAddr and src are meaningful).
    inst_params = DSStoreB192(
        dstAddr=vgpr(5),
        src=vgpr("P+0", 6),
    )
    params = inst_params.getParams()
    # params should have 3 elements: dstAddr, src0, src1(nullptr)
    assert len(params) == 3, \
        f"getParams should return 3 elements, got {len(params)}"
    assert str(params[0]) == "v5", \
        f"params[0] (dstAddr) should be v5, got {str(params[0])}"
    assert "vgprP+0" in str(params[1]), \
        f"params[1] (src0) should contain vgprP+0, got {str(params[1])}"


def test_instruction_ds_load_b192():
      """Tests for the DSLoadB192 compound instruction.
      DSLoadB192 is a 192-bit (6 dword) DS load that decomposes into two
      sub-instructions at assembly time:
        1. ds_read_b128 (lower 4 dwords into dst[0:3])  [ds_load_b128 for ISA >= gfx11]
        2. ds_read_b64  (upper 2 dwords into dst[4:5])   [ds_load_b64  for ISA >= gfx11]
      The second sub-instruction has its DS offset incremented by 16 bytes.
      Named registers (vgpr("name", n)) must be used for the destination operand
      because getArgStr2() manipulates regName->offsets to split the register
      range. Flat-indexed registers (vgpr(int, n)) would crash because their
      regName is nullopt.
      """
      from rocisa.instruction import DSLoadB192
      from rocisa.container import DSModifiers
      from rocisa.code import Module
      from copy import deepcopy
      # =========================================================================
      # Test 1: Basic construction with named registers, no modifiers, no comment
      # =========================================================================
      # dst: named register with 6 dwords (destination for loaded data)
      # src: flat-indexed single VGPR for the LDS address
      inst = DSLoadB192(
          dst=vgpr("DstData+0", 6),  # v[vgprDstData+0:vgprDstData+0+5] -- 6 dwords
          src=vgpr(0),                # v0 -- LDS address register
      )
      output = str(inst)
      # The output should be two lines (two sub-instructions), each ending with \n
      lines = output.strip().split('\n')
      assert len(lines) == 2, \
          f"DSLoadB192 should emit exactly 2 sub-instructions, got {len(lines)}"
      # Line 1: ds_read_b128 loading into the lower 4 dwords of dst
      # ISA (9,0,10) < gfx11, so legacy mnemonic "ds_read_b128" is used
      assert "ds_read_b128" in lines[0], \
          f"First sub-inst should be ds_read_b128, got: {lines[0]}"
      # Lower 4 dwords: vgprDstData+0 with regNum=4 -> v[vgprDstData+0:vgprDstData+0+3]
      assert "vgprDstData+0:vgprDstData+0+3" in lines[0], \
          f"First sub-inst should load into lower 4 dwords, got: {lines[0]}"
      # Source address (v0) should appear
      assert "v0" in lines[0], \
          f"First sub-inst should reference src address v0, got: {lines[0]}"
      # Line 2: ds_read_b64 loading into the upper 2 dwords, offset +16
      assert "ds_read_b64" in lines[1], \
          f"Second sub-inst should be ds_read_b64, got: {lines[1]}"
      # Upper 2 dwords: offset shifted by 4 -> vgprDstData+4, regNum=2
      assert "vgprDstData+4:vgprDstData+4+1" in lines[1], \
          f"Second sub-inst should load into upper 2 dwords, got: {lines[1]}"
      assert "v0" in lines[1], \
          f"Second sub-inst should reference src address v0, got: {lines[1]}"
      # Offset should be 16 (4 dwords * 4 bytes = 16 bytes from base)
      assert "offset:16" in lines[1], \
          f"Second sub-inst should have offset:16, got: {lines[1]}"
      # =========================================================================
      # Test 2: Construction with DS modifiers (explicit offset)
      # =========================================================================
      # When a base offset is provided, the first sub-instruction uses it directly,
      # and the second sub-instruction adds 16 to it.
      inst_with_offset = DSLoadB192(
          dst=vgpr("LoadVal+0", 6),
          src=vgpr(1),
          ds=DSModifiers(offset=64),  # base offset = 64 bytes
      )
      output2 = str(inst_with_offset)
      lines2 = output2.strip().split('\n')
      assert len(lines2) == 2
      # Line 1: should carry the original offset 64
      assert "offset:64" in lines2[0], \
          f"First sub-inst should have offset:64, got: {lines2[0]}"
      # Line 2: should carry offset 64 + 16 = 80
      assert "offset:80" in lines2[1], \
          f"Second sub-inst should have offset:80, got: {lines2[1]}"
      # =========================================================================
      # Test 3: Construction with comment
      # =========================================================================
      # Comments should appear on both sub-instruction lines
      inst_commented = DSLoadB192(
          dst=vgpr("C+0", 6),
          src=vgpr(0),
          comment="load 192b"
      )
      output3 = str(inst_commented)
      lines3 = output3.strip().split('\n')
      assert len(lines3) == 2
      # Both lines should contain the comment
      assert "load 192b" in lines3[0], \
          f"Comment should appear in first line, got: {lines3[0]}"
      assert "load 192b" in lines3[1], \
          f"Comment should appear in second line, got: {lines3[1]}"
      # Verify comment field is accessible and correct
      assert inst_commented.comment == "load 192b"
      # =========================================================================
      # Test 4: Static issueLatency()
      # =========================================================================
      assert DSLoadB192.issueLatency() == 3, \
          f"issueLatency should be 3, got {DSLoadB192.issueLatency()}"
      # =========================================================================
      # Test 5: deepcopy independence
      # =========================================================================
      inst_orig = DSLoadB192(
          dst=vgpr("CopyTest+0", 6),
          src=vgpr(0),
          comment="original"
      )
      inst_copy = deepcopy(inst_orig)
      # The copy should produce identical assembly output
      assert str(inst_orig) == str(inst_copy), \
          "deepcopy should produce identical assembly output"
      # But they should be distinct objects
      assert inst_orig is not inst_copy, \
          "deepcopy should produce a new object"
      # Mutating the original's comment should not affect the copy
      inst_orig.comment = "modified"
      assert inst_copy.comment == "original", \
          "deepcopy should be independent: modifying original must not affect copy"
      # Verify the assembly output diverges after mutation
      assert str(inst_orig) != str(inst_copy), \
          "After comment mutation, assembly outputs should differ"
      # =========================================================================
      # Test 6: Module integration and counting functions
      # =========================================================================
      module = Module("DSLoadB192Test")
      inst_for_module = DSLoadB192(
          dst=vgpr("ModDst+0", 6),
          src=vgpr(0),
      )
      module.add(inst_for_module)
      # countLocalRead counts any LocalReadInstruction-derived instruction
      assert rocisa.countLocalRead(module) == 1, \
          f"countLocalRead should be 1, got {rocisa.countLocalRead(module)}"
      # countWeightedLocalRead gives weight 2 for DSLoadB192 because it
      # decomposes into 2 sub-instructions at assembly time
      assert rocisa.countWeightedLocalRead(module) == 2, \
          f"countWeightedLocalRead should be 2, got {rocisa.countWeightedLocalRead(module)}"
      # =========================================================================
      # Test 7: getParams() verification
      # =========================================================================
      # DSLoadInstruction::getParams() returns {dst, srcs} (2 elements).
      inst_params = DSLoadB192(
          dst=vgpr("Q+0", 6),
          src=vgpr(3),
      )
      params = inst_params.getParams()
      assert len(params) == 2, \
          f"getParams should return 2 elements, got {len(params)}"
      # params[0] is dst (the named destination register)
      assert "vgprQ+0" in str(params[0]), \
          f"params[0] (dst) should contain vgprQ+0, got {str(params[0])}"
      # params[1] is srcs (the source address register)
      assert str(params[1]) == "v3", \
          f"params[1] (srcs) should be v3, got {str(params[1])}"


if __name__ == "__main__":
    test_instruction_common()
    test_instruction_cvt()
    test_instruction_tdm()
    test_instruction_tdm_2_sgprs()
    test_instruction_ds_store_b192()
    test_instruction_ds_load_b192()

