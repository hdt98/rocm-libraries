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

"""Tests for the source map between assembly and Python rocasm lines."""

from pathlib import Path

import pytest

from rocasm.asm_to_rocasm import asm_to_rocasm
from rocasm.setparse import parse_set_directives
from rocasm.source_map import MainloopSourceMap, SourceMapping


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MAINLOOP_ASM = FIXTURE_DIR / "mainloop_128x256x64.s"
SET_DIRECTIVES = FIXTURE_DIR / "set_directives_128x256x64.s"

pytestmark = pytest.mark.skipif(
    not MAINLOOP_ASM.exists(),
    reason=f"Assembly fixture not found: {MAINLOOP_ASM}",
)


@pytest.fixture
def source_map():
    """Build a source map from the real 128x256x64 mainloop assembly."""
    symbols = parse_set_directives(SET_DIRECTIVES.read_text())
    asm_text = MAINLOOP_ASM.read_text()
    _, _, _, sm = asm_to_rocasm(asm_text, symbols)
    return MainloopSourceMap(entries=sm)


class TestSourceMapFromRealKernel:

    def test_source_map_length(self, source_map):
        """Source map has one entry per assembly line (216 = 1 label + 215 instructions)."""
        assert len(source_map) == 216

    def test_source_map_first_entry(self, source_map):
        """First entry maps label assembly line to label() Python call."""
        first = source_map.entries[0]
        assert "label_LoopBeginL" in first.asm_text
        assert first.python_text == 'label("label_LoopBeginL")'
        assert first.asm_index == 0
        assert first.python_index == 0

    def test_source_map_mfma_entry(self, source_map):
        """An MFMA assembly line maps to vmfma Python call."""
        # Second entry should be s_waitcnt, third should be first MFMA
        mfma_entry = source_map.entries[2]
        assert "v_mfma_f32_16x16x16bf16_1k" in mfma_entry.asm_text
        assert "vmfma_f32_16x16x16bf16_1k" in mfma_entry.python_text

    def test_source_map_round_trip(self, source_map):
        """Mapping is bijective: asm_to_python and python_to_asm are inverses."""
        for i in range(len(source_map)):
            assert source_map.asm_to_python(i).asm_index == i
            assert source_map.python_to_asm(i).python_index == i

    def test_source_map_asm_text_is_original(self, source_map):
        """asm_text preserves original assembly (with symbolic register names)."""
        # Find an MFMA entry — it should have vgprValuB or vgprValuA
        mfma_entries = [e for e in source_map.entries
                        if "v_mfma" in e.asm_text]
        assert len(mfma_entries) > 0
        # At least some should have symbolic names (from .set resolution context)
        has_symbolic = any("vgpr" in e.asm_text for e in mfma_entries)
        assert has_symbolic, "Expected some MFMA entries to have symbolic register names"

    def test_no_unhandled_entries(self, source_map):
        """No UNHANDLED entries in the source map."""
        for e in source_map.entries:
            assert "UNHANDLED" not in e.python_text


class TestSourceMapSerialization:

    def test_json_round_trip(self, source_map, tmp_path):
        """Serialize to JSON and deserialize, verify round-trip."""
        json_path = tmp_path / "test.map.json"
        source_map.to_json(json_path)
        loaded = MainloopSourceMap.from_json(json_path)
        assert len(loaded) == len(source_map)
        for orig, loaded_entry in zip(source_map.entries, loaded.entries):
            assert orig.asm_index == loaded_entry.asm_index
            assert orig.asm_text == loaded_entry.asm_text
            assert orig.python_index == loaded_entry.python_index
            assert orig.python_text == loaded_entry.python_text


class TestConverterReturnsSourceMap:

    def test_converter_returns_source_map(self):
        """asm_to_rocasm returns source map as 4th element."""
        asm = "s_waitcnt lgkmcnt(0)\ns_barrier"
        _, _, _, sm = asm_to_rocasm(asm, {})
        assert len(sm) == 2
        assert isinstance(sm[0], SourceMapping)
        assert sm[0].python_text == "s_waitcnt(dscnt=0)"
        assert sm[1].python_text == "s_barrier()"
