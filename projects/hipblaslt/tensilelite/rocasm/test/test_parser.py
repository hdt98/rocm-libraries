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

"""Tests for the .set directive parser."""

import pytest
from textwrap import dedent

from rocasm.setparse import parse_set_directives


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
