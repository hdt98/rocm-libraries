################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Tests for the .set directive parser and assembly-to-rocasm converter."""

import pytest
from textwrap import dedent

from rocasm.setparse import parse_set_directives
from rocasm.asm_to_rocasm import (
    asm_to_rocasm,
    parse_register_counts,
    _resolve_expr,
    _extract_base_symbol,
    _make_alias,
    _parse_reg_operand,
    _parse_scalar_operand,
    _split_operands,
    _strip_comment,
)


class TestParseSetDirectives:

    def test_literal_decimal(self):
        result = parse_set_directives(".set vgprBase, 16")
        assert result == {"vgprBase": 16}

    def test_literal_hex(self):
        result = parse_set_directives(".set BufferLimit, 0xffffffff")
        assert result == {"BufferLimit": 0xffffffff}

    def test_literal_hex_short(self):
        result = parse_set_directives(".set Srd127_96, 0x20000")
        assert result == {"Srd127_96": 0x20000}

    def test_symbol_plus_offset(self):
        text = dedent("""\
            .set vgprBase, 16
            .set vgprValuA_X0_I0_BASE, vgprBase+0
        """)
        result = parse_set_directives(text)
        assert result["vgprValuA_X0_I0_BASE"] == 16

    def test_symbol_plus_nonzero_offset(self):
        text = dedent("""\
            .set vgprBase, 16
            .set vgprValuB_X0_I0_BASE, vgprBase+48
        """)
        result = parse_set_directives(text)
        assert result["vgprValuB_X0_I0_BASE"] == 64

    def test_chained_references(self):
        text = dedent("""\
            .set vgprBase, 16
            .set vgprValuA_X0_I0_BASE, vgprBase+0
            .set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+24
        """)
        result = parse_set_directives(text)
        assert result["vgprValuA_X1_I0"] == 40

    def test_symbol_minus_offset(self):
        text = dedent("""\
            .set base, 100
            .set adjusted, base-10
        """)
        result = parse_set_directives(text)
        assert result["adjusted"] == 90

    def test_bare_symbol_alias(self):
        text = dedent("""\
            .set sgprSizesFree, 24
            .set sgprSizeI, sgprSizesFree
        """)
        result = parse_set_directives(text)
        assert result["sgprSizeI"] == 24

    def test_undef_removes_symbol(self):
        text = dedent("""\
            .set sgprSrdA, 64
            .set sgprSrdA, UNDEF
        """)
        result = parse_set_directives(text)
        assert "sgprSrdA" not in result

    def test_undef_nonexistent_is_noop(self):
        result = parse_set_directives(".set foo, UNDEF")
        assert "foo" not in result

    def test_redefinition(self):
        text = dedent("""\
            .set vgprValuA_X0_I0_BASE, 4
            .set vgprValuA_X0_I0_BASE, 16
        """)
        result = parse_set_directives(text)
        assert result["vgprValuA_X0_I0_BASE"] == 16

    def test_non_set_lines_ignored(self):
        text = dedent("""\
            ; this is a comment
            .set vgprBase, 16
            v_mfma_f32_16x16x32_bf16 acc[0:3], v[0:3], v[24:27], acc[0:3]
            .set vgprValuA, vgprBase+0
            label_LoopBeginL:
        """)
        result = parse_set_directives(text)
        assert result == {"vgprBase": 16, "vgprValuA": 16}

    def test_undefined_reference_raises(self):
        with pytest.raises(ValueError, match="undefined symbol 'noSuchSymbol'"):
            parse_set_directives(".set foo, noSuchSymbol+4")

    def test_undefined_bare_reference_raises(self):
        with pytest.raises(ValueError, match="undefined symbol 'noSuchSymbol'"):
            parse_set_directives(".set foo, noSuchSymbol")

    def test_empty_input(self):
        assert parse_set_directives("") == {}

    def test_leading_whitespace(self):
        result = parse_set_directives("   .set vgprBase, 16")
        assert result == {"vgprBase": 16}

    def test_real_kernel_snippet(self):
        """Parse a snippet from a real generated GEMM kernel .s file."""
        text = dedent("""\
            .set vgprValuC, 0
            .set vgprBase, 16
            .set vgprGlobalReadOffsetA, 0
            .set vgprGlobalReadOffsetB, 6
            .set vgprLocalReadAddrA, 14
            .set vgprLocalReadAddrB, 15
            .set vgprSerial, 128
            .set vgprValuA_X0_I0_BASE, vgprBase+0
            .set vgprValuB_X0_I0_BASE, vgprBase+48
            .set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
            .set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+24
            .set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
            .set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+32
            .set sgprKernArgAddress, 0
            .set sgprSrdA, 64
            .set sgprSizesFree, 20
            .set sgprSizeI, sgprSizesFree+0
            .set sgprSizeJ, sgprSizesFree+1
            .set sgprSizeK, sgprSizesFree+2
            .set MT0, 192
            .set MT1, 256
            .set DepthU, 64
            .set BufferLimit, 0xffffffff
        """)
        result = parse_set_directives(text)

        assert result["vgprBase"] == 16
        assert result["vgprValuA_X0_I0_BASE"] == 16
        assert result["vgprValuB_X0_I0_BASE"] == 64
        assert result["vgprValuA_X0_I0"] == 16
        assert result["vgprValuA_X1_I0"] == 40
        assert result["vgprValuB_X0_I0"] == 64
        assert result["vgprValuB_X1_I0"] == 96
        assert result["sgprSrdA"] == 64
        assert result["sgprSizeI"] == 20
        assert result["sgprSizeJ"] == 21
        assert result["sgprSizeK"] == 22
        assert result["MT0"] == 192
        assert result["MT1"] == 256
        assert result["DepthU"] == 64
        assert result["BufferLimit"] == 0xffffffff

    def test_undef_then_redefine(self):
        """Symbols can be UNDEF'd and then redefined later."""
        text = dedent("""\
            .set sgprSrdA, 52
            .set sgprSrdA, UNDEF
            .set sgprSrdA, 64
        """)
        result = parse_set_directives(text)
        assert result["sgprSrdA"] == 64


# ─── asm_to_rocasm helpers ───────────────────────────────────────────────────


class TestResolveExpr:

    def test_literal_int(self):
        assert _resolve_expr("42", {}) == 42

    def test_symbol(self):
        assert _resolve_expr("vgprBase", {"vgprBase": 16}) == 16

    def test_symbol_plus_offset(self):
        assert _resolve_expr("vgprBase+4", {"vgprBase": 16}) == 20

    def test_chained_adds(self):
        assert _resolve_expr("vgprValuA_X0_I0+4+0+0", {"vgprValuA_X0_I0": 16}) == 20

    def test_symbol_plus_offset_plus_count(self):
        assert _resolve_expr("vgprValuA_X0_I0+4+0+0+3", {"vgprValuA_X0_I0": 16}) == 23

    def test_undefined_symbol_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_expr("noSuch+4", {})


class TestExtractBaseSymbol:

    def test_pure_literal(self):
        assert _extract_base_symbol("0", {}) == (None, 0)

    def test_literal_int(self):
        assert _extract_base_symbol("42", {}) == (None, 42)

    def test_symbol_only(self):
        syms = {"vgprValuA_X0_I0": 16}
        assert _extract_base_symbol("vgprValuA_X0_I0", syms) == ("vgprValuA_X0_I0", 0)

    def test_symbol_plus_offset(self):
        syms = {"vgprValuA_X0_I0": 16}
        assert _extract_base_symbol("vgprValuA_X0_I0+4", syms) == ("vgprValuA_X0_I0", 4)

    def test_symbol_plus_chained_zeros(self):
        syms = {"vgprValuA_X0_I0": 16}
        assert _extract_base_symbol("vgprValuA_X0_I0+4+0+0", syms) == ("vgprValuA_X0_I0", 4)

    def test_symbol_plus_chained_with_end(self):
        syms = {"vgprValuA_X0_I0": 16}
        assert _extract_base_symbol("vgprValuA_X0_I0+4+0+0+3", syms) == ("vgprValuA_X0_I0", 7)


class TestMakeAlias:

    def test_valu_a_compact(self):
        assert _make_alias("vgprValuA_X0_I0") == "A0"

    def test_valu_a_x1(self):
        assert _make_alias("vgprValuA_X1_I0") == "A1"

    def test_valu_b_compact(self):
        assert _make_alias("vgprValuB_X0_I0") == "B0"

    def test_valu_b_x1(self):
        assert _make_alias("vgprValuB_X1_I0") == "B1"

    def test_sgpr_prefix(self):
        assert _make_alias("sgprSrdA") == "SrdA"

    def test_vgpr_other(self):
        assert _make_alias("vgprLocalReadAddrA") == "LocalReadAddrA"

    def test_no_prefix(self):
        assert _make_alias("BufferLimit") == "BufferLimit"


class TestParseRegOperand:

    def test_acc_range(self):
        assert _parse_reg_operand("acc[0:3]", {}) == ("acc", 0, 4, None, 0)

    def test_v_range_with_symbols(self):
        syms = {"vgprValuA": 16}
        result = _parse_reg_operand("v[vgprValuA+0:vgprValuA+0+3]", syms)
        assert result == ("v", 16, 4, "vgprValuA", 0)

    def test_s_range_with_symbols(self):
        syms = {"sgprSrdA": 64}
        result = _parse_reg_operand("s[sgprSrdA:sgprSrdA+3]", syms)
        assert result == ("s", 64, 4, "sgprSrdA", 0)

    def test_single_vgpr(self):
        syms = {"vgprLocalReadAddrA": 14}
        result = _parse_reg_operand("v[vgprLocalReadAddrA]", syms)
        assert result == ("v", 14, 1, "vgprLocalReadAddrA", 0)

    def test_v_range_with_offset(self):
        syms = {"vgprValuA_X0_I0": 16}
        result = _parse_reg_operand("v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3]", syms)
        assert result == ("v", 20, 4, "vgprValuA_X0_I0", 4)

    def test_not_a_register(self):
        assert _parse_reg_operand("0x10000", {}) is None
        assert _parse_reg_operand("m0", {}) is None


class TestParseScalarOperand:

    def test_sgpr_single(self):
        assert _parse_scalar_operand("s[sgprSrdA+0]", {"sgprSrdA": 64}) == "sgpr(64)"

    def test_sgpr_range(self):
        result = _parse_scalar_operand("s[sgprSrdA:sgprSrdA+3]", {"sgprSrdA": 64})
        assert result == "sgpr(64, 4)"

    def test_bare_s_register(self):
        assert _parse_scalar_operand("s60", {}) == "sgpr(60)"

    def test_vgpr_single(self):
        assert _parse_scalar_operand("v[vgprLocalReadAddrA]", {"vgprLocalReadAddrA": 14}) == "vgpr(14)"

    def test_m0(self):
        assert _parse_scalar_operand("m0", {}) == "m0"

    def test_hex_literal(self):
        assert _parse_scalar_operand("0x10000", {}) == "0x10000"

    def test_decimal_literal(self):
        assert _parse_scalar_operand("42", {}) == "42"

    def test_symbol_reference(self):
        assert _parse_scalar_operand("BufferLimit", {"BufferLimit": 0xffffffff}) == "4294967295"


class TestSplitOperands:

    def test_simple(self):
        assert _split_operands("a, b, c") == ["a", "b", "c"]

    def test_brackets(self):
        result = _split_operands("v[a+0:a+3], s[b:b+3], 0")
        assert result == ["v[a+0:a+3]", "s[b:b+3]", "0"]

    def test_single(self):
        assert _split_operands("abc") == ["abc"]


class TestStripComment:

    def test_with_comment(self):
        assert _strip_comment("s_barrier // sync") == "s_barrier"

    def test_no_comment(self):
        assert _strip_comment("s_barrier") == "s_barrier"


class TestParseRegisterCounts:

    def test_parse_all_three(self):
        text = dedent("""\
            /* Num VGPR   =256 */
            /* Num AccVGPR=192 */
            /* Num SGPR   =88 */
        """)
        result = parse_register_counts(text)
        assert result == {"vgpr": 256, "accvgpr": 192, "sgpr": 88}

    def test_different_spacing(self):
        text = "/* Num VGPR=128 */\n/* Num AccVGPR =64 */\n/* Num SGPR= 32 */"
        result = parse_register_counts(text)
        assert result == {"vgpr": 128, "accvgpr": 64, "sgpr": 32}

    def test_missing_counts_returns_partial(self):
        result = parse_register_counts("/* Num VGPR =256 */")
        assert result == {"vgpr": 256}
        assert "accvgpr" not in result

    def test_no_matches(self):
        assert parse_register_counts("no register info here") == {}


# ─── asm_to_rocasm integration tests ────────────────────────────────────────


def _body(result: tuple | str) -> str:
    """Extract the instruction body from asm_to_rocasm output, skipping the alias preamble."""
    code = result[0] if isinstance(result, tuple) else result
    lines = code.split("\n")
    # Find the blank line separating preamble from body
    for i, line in enumerate(lines):
        if line == "":
            return "\n".join(lines[i + 1:])
    # No preamble (e.g. only unhandled lines) — return as-is
    return code


def _code(result: tuple[str, dict]) -> str:
    """Extract just the code string from the asm_to_rocasm tuple return."""
    return result[0]


class TestAsmToRocasm:

    SYMBOLS = {
        "vgprValuA_X0_I0": 16,
        "vgprValuA_X1_I0": 40,
        "vgprValuB_X0_I0": 64,
        "vgprValuB_X1_I0": 96,
        "vgprLocalReadAddrA": 14,
        "vgprLocalReadAddrB": 15,
        "sgprSrdA": 64,
        "sgprSrdB": 68,
        "sgprLoopCounterL": 8,
        "sgprStaggerUIter": 76,
        "sgprWrapUA": 77,
        "sgprGlobalReadIncsA": 81,
        "sgprShadowLimitA": 72,
        "sgprLocalWriteAddrA": 53,
        "sgprLocalWriteAddrB": 54,
        "BufferLimit": 0xFFFFFFFF,
        "vgprGlobalReadOffsetA": 0,
        "vgprLocalWriteAddrA": 186,
        "vgprLocalWriteAddrB": 187,
        "vgprG2LA": 130,
        "vgprG2LB": 154,
    }

    def test_preamble_has_register_aliases(self):
        asm = "v_mfma_f32_16x16x32_bf16 acc[0:3], v[64:67], v[16:19], acc[0:3]"
        code, named_regs, *_ = asm_to_rocasm(asm, {})
        assert "Acc = block.Acc" in code
        assert "V = block.V" in code

    def test_preamble_has_method_aliases(self):
        asm = "s_waitcnt lgkmcnt(0)"
        code, *_ = asm_to_rocasm(asm, {})
        assert "s_waitcnt = block.s_waitcnt" in code

    def test_mfma(self):
        asm = "v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3]"
        result = asm_to_rocasm(asm, self.SYMBOLS)
        body = _body(result)
        assert body == (
            "Acc[0:4] = vmfma_f32_16x16x32_bf16("
            "B0[0:4], A0[0:4], Acc[0:4])"
        )

    def test_mfma_named_regs(self):
        asm = "v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3]"
        _, named_regs, *_ = asm_to_rocasm(asm, self.SYMBOLS)
        assert "B0" in named_regs
        assert "A0" in named_regs
        assert named_regs["B0"] == ("v", 64, 4)
        assert named_regs["A0"] == ("v", 16, 4)

    def test_ds_read_b128_with_offset(self):
        asm = "ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:64"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == (
            "A1[0:4] = ds_read_b128("
            "LocalReadAddrA[0:1], ds=DSModifiers(offset=64))"
        )

    def test_ds_read_b128_no_offset(self):
        asm = "ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == "A0[0:4] = ds_read_b128(LocalReadAddrA[0:1])"

    def test_buffer_load_lds(self):
        asm = "buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert "buffer_load_lds(" in body
        assert "GlobalReadOffsetA[0:1]" in body
        assert "SrdA[0:4]" in body
        assert "offen=True" in body
        assert "lds=True" in body

    def test_s_waitcnt_lgkmcnt(self):
        body = _body(asm_to_rocasm("s_waitcnt lgkmcnt(7)", self.SYMBOLS))
        assert body == "s_waitcnt(dscnt=7)"

    def test_s_waitcnt_vmcnt(self):
        body = _body(asm_to_rocasm("s_waitcnt vmcnt(14)", self.SYMBOLS))
        assert body == "s_waitcnt(vlcnt=14)"

    def test_s_barrier(self):
        body = _body(asm_to_rocasm("s_barrier", self.SYMBOLS))
        assert body == "s_barrier()"

    def test_s_nop(self):
        body = _body(asm_to_rocasm("s_nop 0", self.SYMBOLS))
        assert body == "s_nop(0)"

    def test_s_mov_b32(self):
        body = _body(asm_to_rocasm("s_mov_b32 m0, s[sgprLocalWriteAddrA]", self.SYMBOLS))
        assert body == "s_mov_b32(dst=m0, src=sgpr(53))"

    def test_s_add_u32(self):
        body = _body(asm_to_rocasm("s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60", self.SYMBOLS))
        assert body == "s_add_u32(dst=sgpr(64), src0=sgpr(64), src1=sgpr(60))"

    def test_s_add_u32_m0(self):
        body = _body(asm_to_rocasm("s_add_u32 m0, m0, 4352", self.SYMBOLS))
        assert body == "s_add_u32(dst=m0, src0=m0, src1=4352)"

    def test_s_cmp_eq_u32(self):
        body = _body(asm_to_rocasm("s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]", self.SYMBOLS))
        assert body == "s_cmp_eq_u32(src0=sgpr(8), src1=sgpr(76))"

    def test_s_cmp_eq_i32(self):
        body = _body(asm_to_rocasm("s_cmp_eq_i32 s[sgprLoopCounterL], 0x2", self.SYMBOLS))
        assert body == "s_cmp_eq_i32(src0=sgpr(8), src1=0x2)"

    def test_s_cselect_b32(self):
        body = _body(asm_to_rocasm(
            "s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit",
            self.SYMBOLS,
        ))
        assert body == "s_cselect_b32(dst=sgpr(66), src0=sgpr(72), src1=4294967295)"

    def test_s_cbranch_scc0(self):
        body = _body(asm_to_rocasm("s_cbranch_scc0 label_LoopBeginL", self.SYMBOLS))
        assert body == 's_cbranch_scc0(labelName="label_LoopBeginL")'

    def test_v_xor_b32(self):
        body = _body(asm_to_rocasm(
            "v_xor_b32 v[vgprLocalReadAddrA], 0x10000, v[vgprLocalReadAddrA]",
            self.SYMBOLS,
        ))
        assert body == "v_xor_b32(dst=vgpr(14), src0=0x10000, src1=vgpr(14))"

    def test_s_xor_b32(self):
        body = _body(asm_to_rocasm(
            "s_xor_b32 s[sgprLocalWriteAddrA], 0x10000, s[sgprLocalWriteAddrA]",
            self.SYMBOLS,
        ))
        assert body == "s_xor_b32(dst=sgpr(53), src0=0x10000, src1=sgpr(53))"

    def test_label(self):
        body = _body(asm_to_rocasm("label_LoopBeginL:", self.SYMBOLS))
        assert body == 'label("label_LoopBeginL")'

    def test_comments_skipped(self):
        asm = dedent("""\
            // this is a comment
            /* block comment */
            s_barrier
        """)
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == "s_barrier()"

    def test_trailing_comment_stripped(self):
        body = _body(asm_to_rocasm("s_barrier // sync threads", self.SYMBOLS))
        assert body == "s_barrier()"

    def test_set_directives_skipped(self):
        asm = dedent("""\
            .set vgprBase, 16
            s_barrier
        """)
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == "s_barrier()"

    def test_unhandled_instruction(self):
        code, *_ = asm_to_rocasm("v_unknown_op v0, v1, v2", self.SYMBOLS)
        assert "# UNHANDLED:" in code

    def test_multi_instruction_snippet(self):
        """Verify a realistic multi-instruction sequence produces correct output."""
        asm = dedent("""\
            s_waitcnt lgkmcnt(0)
            s_barrier
            v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // mfma 0
            s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec
            s_cbranch_scc0 label_LoopBeginL
        """)
        code, named_regs, *_ = asm_to_rocasm(asm, self.SYMBOLS)
        body = _body(code)
        lines = body.strip().split("\n")
        assert len(lines) == 5
        assert lines[0] == "s_waitcnt(dscnt=0)"
        assert lines[1] == "s_barrier()"
        assert lines[2].startswith("Acc[0:4] = vmfma")
        assert lines[3] == "s_sub_u32(dst=sgpr(8), src0=sgpr(8), src1=1)"
        assert lines[4] == 's_cbranch_scc0(labelName="label_LoopBeginL")'
        # Check preamble has named + flat aliases
        assert "Acc = block.Acc" in code
        assert "A0 = block.A0" in code
        assert "B0 = block.B0" in code
        assert "s_barrier = block.s_barrier" in code
        assert "s_cbranch_scc0 = block.s_cbranch_scc0" in code
        assert "s_sub_u32 = block.s_sub_u32" in code
        assert "s_waitcnt = block.s_waitcnt" in code
        # Named regs dict should have the compact aliases
        assert "A0" in named_regs
        assert "B0" in named_regs

    def test_symbols_auto_parsed(self):
        """When symbols=None, .set directives in the text are parsed automatically."""
        asm = dedent("""\
            .set myReg, 10
            s_mov_b32 s[myReg], s[myReg]
        """)
        body = _body(asm_to_rocasm(asm))
        assert body == "s_mov_b32(dst=sgpr(10), src=sgpr(10))"

    def test_acc_uses_flat_alias(self):
        """Accumulators without named symbols should use the flat Acc alias."""
        asm = "v_mfma_f32_16x16x32_bf16 acc[0:3], v[64:67], v[16:19], acc[0:3]"
        body = _body(asm_to_rocasm(asm, {}))
        assert "Acc[0:4]" in body
        assert "V[64:68]" in body
        assert "V[16:20]" in body

    def test_mfma_16x16x16bf16_1k(self):
        """gfx942 MFMA variant with 2-VGPR sources."""
        asm = ("v_mfma_f32_16x16x16bf16_1k acc[0:3], "
               "v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], "
               "v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], "
               "acc[0:3]")
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == (
            "Acc[0:4] = vmfma_f32_16x16x16bf16_1k("
            "B0[0:2], A0[0:2], Acc[0:4])")

    def test_ds_write_b128_with_offset(self):
        asm = "ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:4608"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == "ds_write_b128(LocalWriteAddrA[0:1], G2LA[0:4], ds=DSModifiers(offset=4608))"

    def test_ds_write_b128_no_offset(self):
        asm = "ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        # offset:0 should not produce a DSModifiers arg
        assert body == "ds_write_b128(LocalWriteAddrA[0:1], G2LA[0:4])"

    def test_ds_write_b32_with_offset(self):
        asm = "ds_write_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24+0] offset:25344"
        body = _body(asm_to_rocasm(asm, self.SYMBOLS))
        assert body == "ds_write_b32(LocalWriteAddrB[0:1], G2LB[24:25], ds=DSModifiers(offset=25344))"

    def test_inst_funcs_returned(self):
        """The third return element should contain instruction function names."""
        asm = dedent("""\
            v_mfma_f32_16x16x16bf16_1k acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], acc[0:3]
            ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0
        """)
        _, _, inst_funcs, _ = asm_to_rocasm(asm, self.SYMBOLS)
        assert "vmfma_f32_16x16x16bf16_1k" in inst_funcs
        assert "ds_write_b128" in inst_funcs
