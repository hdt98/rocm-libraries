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

"""Tests for the ATT trace parser."""

import json
import os
from pathlib import Path

import pytest

from rocasm.att import ATTProfile, InstructionPerf


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CODE_JSON = FIXTURE_DIR / "att_code_128x256x64.json"

# Skip all tests in this module if the fixture doesn't exist
pytestmark = pytest.mark.skipif(
    not CODE_JSON.exists(),
    reason=f"ATT fixture not found: {CODE_JSON}",
)


class TestATTProfile:

    @pytest.fixture
    def profile(self):
        return ATTProfile.from_code_json(CODE_JSON)

    def test_parse_code_json(self, profile):
        """Parse the real code.json fixture and verify basic structure."""
        assert len(profile.all_instructions) == 2459
        first = profile.all_instructions[0]
        assert isinstance(first, InstructionPerf)
        assert first.isa == "s_nop 0"
        assert first.line == 1

    def test_mainloop_filter(self, profile):
        """Mainloop filter returns steady-state instructions with dominant hitcount."""
        ml = profile.mainloop_instructions()
        assert len(ml) == 215
        hitcounts = set(inst.hitcount for inst in ml)
        assert hitcounts == {248}

    def test_mainloop_first_instruction(self, profile):
        """First mainloop instruction is s_waitcnt."""
        ml = profile.mainloop_instructions()
        assert ml[0].isa.startswith("s_waitcnt")

    def test_mainloop_last_instruction(self, profile):
        """Last mainloop instruction is the branch back."""
        ml = profile.mainloop_instructions()
        assert "s_cbranch_scc0" in ml[-1].isa

    def test_buffer_load_latency(self, profile):
        """Some buffer_load instructions have high latency (visible stalls)."""
        ml = profile.mainloop_instructions()
        buffer_loads = [inst for inst in ml if "buffer_load" in inst.isa]
        assert len(buffer_loads) > 0
        high_lat = [inst for inst in buffer_loads if inst.latency > 1000]
        assert len(high_lat) > 0, "Expected some buffer_loads with high latency"

    def test_perf_fields_non_negative(self, profile):
        """All perf fields should be non-negative integers."""
        for inst in profile.all_instructions:
            assert inst.hitcount >= 0
            assert inst.latency >= 0
            assert inst.stall >= 0
            assert inst.idle >= 0

    def test_from_att_directory(self):
        """from_att_directory navigates the ui_output_agent_* nesting."""
        att_dir = Path("/tmp/att_128x256x64_gfx942")
        if not att_dir.exists():
            pytest.skip("ATT output directory not available")
        profile = ATTProfile.from_att_directory(att_dir)
        assert len(profile.all_instructions) == 2459


class TestATTProfileSynthetic:
    """Tests using synthetic data (no fixture dependency)."""

    def test_empty_profile(self, tmp_path):
        """Empty code list produces empty mainloop."""
        code_json = tmp_path / "code.json"
        code_json.write_text(json.dumps({
            "code": [],
            "header": "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle",
            "version": "3.0.0",
        }))
        profile = ATTProfile.from_code_json(code_json)
        assert len(profile.all_instructions) == 0
        assert profile.mainloop_instructions() == []

    def test_mainloop_dominant_hitcount(self, tmp_path):
        """Mainloop filter picks the hitcount shared by the most instructions."""
        code_json = tmp_path / "code.json"
        entries = []
        # 5 prologue instructions with hit=2
        for i in range(5):
            entries.append([f"s_nop {i}", 0, i+1, "", 1, 1000+i*4, 2, 10, 0, 0])
        # 10 mainloop instructions with hit=100
        for i in range(10):
            entries.append([f"v_mfma {i}", 0, i+6, "", 1, 2000+i*8, 100, 500, 50, 0])
        # 3 epilogue instructions with hit=2
        for i in range(3):
            entries.append([f"s_end {i}", 0, i+16, "", 1, 3000+i*4, 2, 10, 0, 0])
        code_json.write_text(json.dumps({
            "code": entries,
            "header": "...",
            "version": "3.0.0",
        }))
        profile = ATTProfile.from_code_json(code_json)
        ml = profile.mainloop_instructions()
        assert len(ml) == 10
        assert all(inst.hitcount == 100 for inst in ml)
