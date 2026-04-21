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

"""Tests for the profile combiner (ATT perf data + source map)."""

from pathlib import Path

import pytest

from rocasm.att import ATTProfile
from rocasm.asm_to_rocasm import asm_to_rocasm
from rocasm.setparse import parse_set_directives
from rocasm.source_map import MainloopSourceMap
from rocasm.profile import profile_mainloop, ProfiledMainloop


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CODE_JSON = FIXTURE_DIR / "att_code_128x256x64.json"
MAINLOOP_ASM = FIXTURE_DIR / "mainloop_128x256x64.s"
SET_DIRECTIVES = FIXTURE_DIR / "set_directives_128x256x64.s"

pytestmark = pytest.mark.skipif(
    not all(f.exists() for f in [CODE_JSON, MAINLOOP_ASM, SET_DIRECTIVES]),
    reason="Profile test fixtures not found",
)


@pytest.fixture
def profiled() -> ProfiledMainloop:
    """Build a ProfiledMainloop from real ATT trace + real source map."""
    att = ATTProfile.from_code_json(CODE_JSON)
    symbols = parse_set_directives(SET_DIRECTIVES.read_text())
    _, _, _, sm = asm_to_rocasm(MAINLOOP_ASM.read_text(), symbols)
    source_map = MainloopSourceMap(entries=sm)
    return profile_mainloop(att, source_map)


class TestProfiledMainloop:

    def test_profile_length(self, profiled):
        """Profile has 216 lines (1 label + 215 instructions)."""
        assert len(profiled.lines) == 216

    def test_profile_has_perf_data(self, profiled):
        """Non-label lines have hitcount=248 (steady-state)."""
        instruction_lines = [l for l in profiled.lines if l.hitcount > 0]
        assert len(instruction_lines) == 215
        assert all(l.hitcount == 248 for l in instruction_lines)

    def test_label_has_zero_perf(self, profiled):
        """The label line has zero perf data."""
        label = profiled.lines[0]
        assert "label" in label.python_text
        assert label.hitcount == 0
        assert label.latency == 0

    def test_hottest_stall(self, profiled):
        """hottest(5, 'stall') returns 5 entries sorted by stall descending."""
        hot = profiled.hottest(5, key="stall")
        assert len(hot) == 5
        assert all(h.stall > 0 for h in hot)
        stalls = [h.stall for h in hot]
        assert stalls == sorted(stalls, reverse=True)

    def test_buffer_load_stalls(self, profiled):
        """Buffer_load lines visible in screenshot have high stall values."""
        buffer_loads = [l for l in profiled.lines
                        if "buffer_load" in l.python_text]
        assert len(buffer_loads) > 0
        high_stall = [l for l in buffer_loads if l.stall > 1000]
        assert len(high_stall) > 0, "Expected buffer_load lines with high stall"

    def test_python_text_present(self, profiled):
        """Each ProfiledLine has meaningful Python source text."""
        for line in profiled.lines:
            assert len(line.python_text) > 0
        # Verify specific instruction types appear
        python_texts = " ".join(l.python_text for l in profiled.lines)
        assert "vmfma_f32_16x16x16bf16_1k" in python_texts
        assert "buffer_load_dwordx4" in python_texts
        assert "ds_read_b128" in python_texts
        assert "ds_write_b128" in python_texts

    def test_avg_latency(self, profiled):
        """avg_latency = latency / hitcount for lines with hits."""
        for line in profiled.lines:
            if line.hitcount > 0:
                expected = line.latency / line.hitcount
                assert abs(line.avg_latency - expected) < 0.001

    def test_by_instruction_type(self, profiled):
        """Groups contain expected instruction type keys."""
        groups = profiled.by_instruction_type()
        assert "mfma" in groups
        assert "ds_read" in groups
        assert "buffer_load" in groups
        assert "ds_write" in groups
        assert "s_waitcnt" in groups
        # MFMA should be the largest group
        assert len(groups["mfma"]) > len(groups["buffer_load"])
